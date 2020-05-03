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

    if(y - 1 > 0     ) v_in_next[T2I(    x, y - 1, 2)] = v_out_next[0];
    if(x - 1 > 0     ) v_in_next[T2I(x - 1,     y, 3)] = v_out_next[1];
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

#ifdef TEST_TLM
void print_grid(int width, int height, float * v_in) {
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

int main() {
  int width = 8;
  int height = 8;
  float * v_in_curr;
  float * v_in_next;

  tlm_2d_init_zero(width, height, &v_in_curr, &v_in_next);

  printf("Starting...\n");

  int i;
  for(i = 0; i < 64; i++) {
    v_in_curr[T2I(width / 2, height / 2, 0)] = 1; // transmitter

    if(i == 0) {
      printf("Step %d:\n\n", i);
      print_grid(width, height, v_in_curr);
    }

    tlm_2d_step_kernel<<< 1, 1024 >>>(width, height, v_in_curr, v_in_next);
    cudaDeviceSynchronize();

    float * tmp = v_in_curr;
    v_in_curr = v_in_next;
    v_in_next = tmp;
  }

  printf("Step %d:\n\n", i);
  print_grid(width, height, v_in_curr);

  printf("Complete!\n");
}
#endif