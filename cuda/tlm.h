#pragma once

#define _USE_MATH_DEFINES

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

extern "C" {
  __global__ void tlm_2d_step_kernel(int width, int height, float * v_in_curr, float * v_in_next, float * v_max);

  // initialize tlm grid with zeros
  void tlm_2d_init_zero(int width, int height, float** v_in_curr, float ** v_in_next, float ** v_max);

  /*
  // Approximates the linear path loss
  __global__ void path_loss_2d_kernel(
    float P, 
    float dx, 
    int width, 
    int height, 
    float * v_in, 
    float * path_loss
  );
  */

  // https://en.wikipedia.org/wiki/Shannon%E2%80%93Hartley_theorem
  __global__ void channel_capacity_2d_kernel(
    float channel_bandwidth, 
    float noise_power, 
    int width, 
    int height, 
    float * v_max, 
    float * channel_capacity
  );

  // Computes the channel capacity of each client based on its relative distance
  // from the router.
  // 
  // Parameters:
  // n <- number of clients
  // client_x <- array of relative x positions (in range -50 to 50 m)
  // client_y <- array of relative x positions (in range -50 to 50 m)
  // channel_capacity -> pointer to array of channel capacities (bits per second)
  void compute_channel_capacity(int n, float * client_x, float * client_y, float ** channel_capacity);

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
  );
}