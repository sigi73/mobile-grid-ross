#include "tlm.h"

#ifdef TEST_TLM
#include <stdio.h>
#endif

#define T2I(x, y, k) (x) * height * 4 + (y) * 4 + k // index for 2d tlm voltage arrays

__global__ void tlm_2d_step_kernel(int width, int height, float * v_in_curr, float * v_in_next) {
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

    i += blockDim.x*gridDim.x;
  }
}

void tlm_2d_init_zero(int width, int height, float** v_in_curr, float ** v_in_next) {
  cudaMallocManaged(v_in_curr,  width * height * 4 * sizeof(float));
  cudaMallocManaged(v_in_next, width * height * 4 * sizeof(float));

  for(int i = 0; i < width * height * 4; i++) {
    (*v_in_curr)[i] = 0.0;
    (*v_in_next)[i] = 0.0;
  }
}

// Approximates the linear path loss
__global__ void path_loss_2d_kernel(float P, float dx, int width, int height, float * v_in, float * path_loss) {
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

// https://en.wikipedia.org/wiki/Shannon%E2%80%93Hartley_theorem
__global__ void channel_capacity_2d_kernel(
  float channel_bandwidth, 
  float noise_power, 
  int width, 
  int height, 
  float * v_in, 
  float * channel_capacity
) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  while(i < width * height) {  
    int x = i / height;
    int y = i % height;

    float v_max = 0; // approximate signal power at this location
    for(int k = 0; k < 4; k++) {
      v_max = fmax(v_max, fabs(v_in[T2I(x, y, k)]));
    }

    // TODO: Check my math
    channel_capacity[x * height + y] = channel_bandwidth * log2(1 + pow(v_max, 2) / noise_power); // (?) signal power estimated as v_max^2

    i += blockDim.x*gridDim.x;
  }
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

  float P = 0.1; // transmitter power: W
  float dx = 0.1; // grid size: m
  float c = 3e8; // speed of light: m/s
  float f = 2.4e9; // frequency: hz

  float dt = dx / c; // time step: s
  float v0 = sqrt(P)/dx; // transmitter voltage: V (assuming transmitter at constant amperage)

  int width = 8;
  int height = 8;
  float * v_in_curr;
  float * v_in_next;

  tlm_2d_init_zero(width, height, &v_in_curr, &v_in_next);

  printf("Starting...\n");

  int i;
  float t = 0;
  for(i = 0; i < 16384; i++) {
    if (i < 4) {
      v_in_curr[T2I(width / 2, height / 2, 0)] = v0; // transmitter
    }

    if(i == 0) {
      printf("Step %d, t = %E:\n\n", i, t);
      print_tlm_grid(width, height, v_in_curr);
    }

    tlm_2d_step_kernel<<< 1, 64 >>>(width, height, v_in_curr, v_in_next);
    cudaDeviceSynchronize();

    float * tmp = v_in_curr;
    v_in_curr = v_in_next;
    v_in_next = tmp;

    t += dt;
  }

  printf("Step %d, t = %E:\n\n", i, t);
  print_tlm_grid(width, height, v_in_curr);

  float * channel_capacity;
  cudaMallocManaged(&channel_capacity, width * height * sizeof(float));

  channel_capacity_2d_kernel<<< 1, 64 >>>(23e6, 0.1, width, height, v_in_curr, channel_capacity);
  cudaDeviceSynchronize();

  printf("Channel Capacity:\n\n");
  print_grid(width, height, channel_capacity);

  printf("Complete!\n");
}
#endif