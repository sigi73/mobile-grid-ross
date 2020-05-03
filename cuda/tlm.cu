#include "tlm.h"

#ifdef TEST_TLM
#include <stdio.h>
#endif

#define T2I(x, y, k) (x) * height * 4 + (y) * 4 + k // index for 2d tlm voltage arrays

__global__ void tlm_2d_step_kernel(int width, int height, float * v_in_curr, float * v_in_next) {
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

void tlm_2d_init_pulse(int width, int height, float** v_in_curr, float ** v_in_next, float pulse) {
  cudaMallocManaged(v_in_curr,  width * height * 4 * sizeof(float));
  cudaMallocManaged(v_in_next, width * height * 4 * sizeof(float));

  for(int i = 0; i < width * height * 4; i++) {
    (*v_in_curr)[i] = 0.0;
    (*v_in_next)[i] = 0.0;
  }

  (*v_in_curr)[T2I(width / 2, height / 2, 0)] = pulse;
}

#ifdef TEST_TLM
void print_grid(int width, int height, float * v_in) {
  for(int i = 0; i < height * 3; i++) {
    for(int j = 0; j < width * 3; j++) {
      int x = j/3;
      int y = height - (i/3) - 1;

      if (i%3 == 1 && j%3 == 1) {
        printf("[%d, %d]", x, y);
        continue;
      }

      float v;
      int is_space = 1;
      if(i%3 == 0 && j%3 == 1) {
        v =  v_in[T2I(x, y, 2)];
        is_space = 0;
      } else if (i%3 == 1 && j%3 == 0) {
        v =  v_in[T2I(x, y, 1)];
        is_space = 0;
      } else if (i%3 == 1 && j%3 == 2) {
        v =  v_in[T2I(x, y, 3)];
        is_space = 0;
      } else if (i%3 == 2 && j%3 == 1) {
        v =  v_in[T2I(x, y, 0)];
        is_space = 0;
      } 

      if(is_space) {
        printf("      ");
        continue;
      }

      printf("%s%.2f ", v < 0 ? "" : " ", v);
    }
    printf("\n\n");
  } 
}

int main() {
  int width = 8;
  int height = 8;
  float * v_in_curr;
  float * v_in_next;

  tlm_2d_init_pulse(width, height, &v_in_curr, &v_in_next, 1);

  int i;
  for(i = 0; i < 3; i++) {
    printf("Step %d:\n\n", i);
    print_grid(width, height, v_in_curr);

    tlm_2d_step_kernel<<< 1, 32 >>>(width, height, v_in_curr, v_in_next);
    cudaDeviceSynchronize();

    float * tmp = v_in_curr;
    v_in_curr = v_in_next;
    v_in_next = tmp;
  }

  printf("Step %d:\n\n", i);
  print_grid(width, height, v_in_curr);
}
#endif