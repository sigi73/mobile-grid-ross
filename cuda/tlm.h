#pragma once

#define _USE_MATH_DEFINES

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

extern "C" {
  // initialize tlm grid with zeros
  void tlm_2d_init_zero(int width, int height, float** v_in_curr, float ** v_in_next, float ** v_max);

  void alloc_channel_capacity_args(int n, float** client_x, float ** client_y);

  // Computes the channel capacity of each client based on its relative distance
  // from the router.
  // 
  // Parameters:
  // n <- number of clients
  // client_x <- array of relative x positions (in range -50 to 50 m)
  // client_y <- array of relative x positions (in range -50 to 50 m)
  // channel_capacity -> pointer to array of channel capacities (bits per second)
  void compute_channel_capacity(int n, float * client_x, float * client_y, float ** channel_capacity);
}