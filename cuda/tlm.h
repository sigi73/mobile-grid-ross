#pragma once

#include <cuda.h>
#include <cuda_runtime.h>


// v_in/v_out voltages into of each node [width x height x 6], row major order
// dt = dx / (c sqrt 3)
__global__ void tlm_2d_step_kernel(int width, int height, float * v_in_curr, float * v_in_next);

void tlm_2d_init_pulse(int width, int height, float** v_in_curr, float ** v_in_next, float pulse);