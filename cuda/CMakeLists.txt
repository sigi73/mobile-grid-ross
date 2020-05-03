SET(CMAKE_C_COMPILER "nvcc")
SET(CMAKE_CXX_COMPILER "nvcc")

project(mobilegrid-cuda CUDA)

SET(cuda_srcs
    tlm.cu)

ADD_LIBRARY(lchannel_capacity SHARED ${cuda_srcs})
