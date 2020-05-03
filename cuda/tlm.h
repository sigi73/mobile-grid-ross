#pragma once

#include <cuda.h>
#include <cuda_runtime.h>


// v_in/v_out voltages into of each node [width x height x 6], row major order
// dt = dx / c
__global__ void tlm_2d_step_kernel(int width, int height, float * v_in_curr, float * v_in_next);

// initialize tlm grid with zeros
void tlm_2d_init_zero(int width, int height, float** v_in_curr, float ** v_in_next);