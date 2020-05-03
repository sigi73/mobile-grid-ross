IF(MOBILE_GRID_USE_CUDA)
message("In cuda cmake")
SET(CMAKE_C_COMPILER "nvcc")
SET(CMAKE_CXX_COMPILER "nvcc")

project(mobilegrid-cuda CUDA)

SET(cuda_srcs
    tlm.cu)

ADD_LIBRARY(lchannel_capacity SHARED ${cuda_srcs})

message("End cuda cmake2")
ENDIF(MOBILE_GRID_USE_CUDA)