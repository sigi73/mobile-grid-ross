#include <cuda.h>
#include <cuda_runtime.h>
#include "tlm.h"

#include <stdio.h>
#include <stdlib.h> 

#define T2I(x, y, k) (x) * height * 4 + (y) * 4 + k // index for 2d tlm voltage arrays

void tlm_init(int mpi_rank) {
  // Determine number of cuda devices and set device accordingly
  int cE, cudaDeviceCount;
  if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess ) {
    printf(" Unable to determine cuda device count, error is %d, count is %d\n",
    cE, cudaDeviceCount );
    exit(-1);
  }
  if( (cE = cudaSetDevice( mpi_rank % cudaDeviceCount )) != cudaSuccess ) {
    printf(" Unable to have rank %d set to cuda device %d, error is %d \n",
    mpi_rank, (mpi_rank % cudaDeviceCount), cE);
    exit(-1);
  }
}

// v_in/v_out voltages into of each node [width x height x 6], row major order
// dt = dx / c
__global__ void tlm_2d_step_kernel(int width, int height, float * v_in_curr, float * v_in_next, float * v_max) {
  // TODO: implement boundary conditions
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while(i < width * height) {  
    int x = i / height;
    int y = i % height;

    float v = 0;
    for(int k = 0; k < 4; k++) {
      v += 1.0/2.0 * v_in_curr[T2I(x, y, k)];
    }

    float v_out_next[4];
    for(int k = 0; k < 4; k++) {
      v_out_next[k] = v - v_in_curr[T2I(x, y, k)];
    }

    if(y - 1 >= 0    ) v_in_next[T2I(    x, y - 1, 2)] = v_out_next[0];
    if(x - 1 >= 0    ) v_in_next[T2I(x - 1,     y, 3)] = v_out_next[1];
    if(y + 1 < height) v_in_next[T2I(    x, y + 1, 0)] = v_out_next[2];
    if(x + 1 < width ) v_in_next[T2I(x + 1,     y, 1)] = v_out_next[3];

    v_max[x * height + y] = fmax(v_max[x * height + y], fabs(v));

    i += blockDim.x*gridDim.x;
  }
}

void tlm_2d_init_zero(int width, int height, float** v_in_curr, float ** v_in_next, float ** v_max) {
  cudaMallocManaged(v_in_curr,  width * height * 4 * sizeof(float));
  cudaMallocManaged(v_in_next, width * height * 4 * sizeof(float));
  cudaMallocManaged(v_max, width * height * sizeof(float));

  for(int i = 0; i < width * height * 4; i++) {
    (*v_in_curr)[i] = 0.0;
    (*v_in_next)[i] = 0.0;
  }

  for(int i = 0; i < width * height; i++) {
    (*v_max)[i] = 0.0;
  }
}

/*
// Approximates the linear path loss
__global__ void path_loss_2d_kernel(
  float P, 
  float dx, 
  int width, 
  int height, 
  float * v_in, 
  float * path_loss
) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  while(i < width * height) {  
    int x = i / height;
    int y = i % height;

    float v_max = 0;
    for(int k = 0; k < 4; k++) {
      v_max = fmax(v_max, fabs(v_in[T2I(x, y, k)]));
    }

    // TODO: Check my math
    path_loss[x * height + y] = (P / pow(dx, 2)) / pow(v_max, 2); // (?) path loss approximated as v0^2/v_max^2

    i += blockDim.x*gridDim.x;
  }
}
*/

// https://en.wikipedia.org/wiki/Shannon%E2%80%93Hartley_theorem
__global__ void channel_capacity_2d_kernel(
  float channel_bandwidth, 
  float noise_power, 
  int width, 
  int height, 
  float * v_max, 
  float * channel_capacity
) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  while(i < width * height) {  

    // TODO: Check my math
    channel_capacity[i] = channel_bandwidth * log2(1 + pow(v_max[i], 2) / noise_power); // (?) signal power estimated as v_max^2

    i += blockDim.x*gridDim.x;
  }
}

// Discretizes xs/ys into lattice points based on bounds
// height) and gets the values from the grid at the specified points
// Sets point to default value if out of bounds
__global__ void get_grid_values_kernel(
  float bounds_left, 
  float bounds_bottom, 
  float bounds_width, 
  float bounds_height, 
  float default_value,
  int grid_width,
  int grid_height,
  float* grid,
  int n,
  float* xs,
  float* ys,
  float* values
) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  while(i < n) {
    int x = round(grid_width * (xs[i] - bounds_left) / bounds_width);
    int y = round(grid_height * (ys[i] - bounds_bottom) / bounds_height);

    if(x < 0 || x >= grid_width || y < 0 || y >= grid_height) {
      values[i] = default_value;
    } else {
      values[i] = grid[x * grid_height + y];
    }

    i += blockDim.x*gridDim.x;
  }  
}

void alloc_channel_capacity_args(int n, float** client_x, float ** client_y, float ** channel_capacity) {
  cudaMallocManaged(client_x, n * sizeof(float));  
  cudaMallocManaged(client_y, n * sizeof(float));  
  cudaMallocManaged(channel_capacity, n * sizeof(float));  
}

void free_channel_capacity_args(float** client_x, float ** client_y, float ** channel_capacity) {
  cudaFree(client_x);  
  cudaFree(client_y);
  cudaFree(channel_capacity);
}

void compute_channel_capacity(
  int n, 
  float * client_x, 
  float * client_y, 
  float * channel_capacity
) {
    //////////////////////
    ////  PARAMETERS  ////
    //////////////////////

    float P = 0.1; // transmitter power: W
    float dx = 0.1; // grid size: m

    int size = 1024; // size of tlm, grid
    float bandwidth = 23e6; // wifi bandwidth in hz
    float noise_power = 0.1; // noise power in V^2

    int steps = 8192; // steps of the simulation to run

    int blocks = 256;
    int threads = 1024;

    //////////////////////
    ////  SIMULATION  ////
    //////////////////////
    
    float * v_in_curr;
    float * v_in_next;
    float * v_max;

    tlm_2d_init_zero(size, size, &v_in_curr, &v_in_next, &v_max);

    float v0 = sqrt(P)/dx; // transmitter voltage: V (assuming transmitter at constant amperage)
    int height = size; // T2I macro needs this defined to work

    for(int i = 0; i < steps; i++) {
      v_in_curr[T2I(size / 2, size / 2, 0)] = v0; // transmitter

      tlm_2d_step_kernel<<< blocks, threads >>>(size, size, v_in_curr, v_in_next, v_max);
      cudaDeviceSynchronize();

      float * tmp = v_in_curr;
      v_in_curr = v_in_next;
      v_in_next = tmp;
    }

    float * channel_capacity_grid;
    cudaMallocManaged(&channel_capacity_grid, size * size * sizeof(float));

    channel_capacity_2d_kernel<<< blocks, threads >>>(bandwidth, noise_power, size, size, v_max, channel_capacity_grid);
    cudaDeviceSynchronize();

    //////////////////////////
    ////  EXTRACT VALUES  ////
    //////////////////////////

    get_grid_values_kernel<<< blocks, threads >>>(
      -dx * size / 2, 
      -dx * size / 2, 
      dx * size, 
      dx * size,
      0,
      size,
      size,
      channel_capacity_grid,
      n,
      client_x,
      client_y,
      channel_capacity
    );

    cudaDeviceSynchronize();

    cudaFree(channel_capacity_grid);
}

#ifdef TEST_TLM
void print_tlm_grid(int width, int height, float * v_in) {
  for(int i = 0; i < height * 3; i++) {
    for(int j = 0; j < width * 3; j++) {
      int x = j/3;
      int y = height - (i/3) - 1;

      if(i%3 == 0 && j%3 == 1) {
        printf("%5.2f ", v_in[T2I(x, y, 2)]);
      } else if (i%3 == 1 && j%3 == 0) {
        printf("%5.2f ", v_in[T2I(x, y, 1)]);
      } else if (i%3 == 1 && j%3 == 2) {
        printf("%5.2f ", v_in[T2I(x, y, 3)]);
      } else if (i%3 == 2 && j%3 == 1) {
        printf("%5.2f ", v_in[T2I(x, y, 0)]);
      } else if (i%3 == 1 && j%3 == 1) {
        printf("[%d, %d]", x, y);
      } else {
        printf("      ");
      }
    }
    printf("\n\n");
  }
}

void print_grid(int width, int height, float * v) {
  for(int y = height - 1; y >= 0; y--) {
    for(int x = 0; x < width; x++) {
      printf("%8.2E ", v[x * height + y]);
    }
    printf("\n\n");
  }
}

int main() {
  const int n = 1024;
  float* client_x;
  float* client_y;
  float* channel_capacity;
  
  alloc_channel_capacity_args(n, &client_x, &client_y, &channel_capacity);

  for(int i = 0; i < n; i++) {
    client_x[i] = ((float) rand() / RAND_MAX) * 50 - 25;
    client_y[i] = ((float) rand() / RAND_MAX) * 50 - 25;
  }

  compute_channel_capacity(n, client_x, client_y, channel_capacity);

  for(int i = 0; i < n; i++) {
    printf("x: %6.2f y: %6.2f cap: %E\n", client_x[i], client_y[i], channel_capacity[i]);
  }
}
#endif