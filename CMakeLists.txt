INCLUDE_DIRECTORIES(${ROSS_SOURCE_DIR} ${ROSS_BINARY_DIR})
## Must include binary dir for config.h

## If you will be using the RIO library add this
#INCLUDE_DIRECTORIES(${RIO_SOURCE_DIR} ${RIO_BINARY_DIR})

## Model Version
LIST(APPEND CMAKE_MODULE_PATH "${ROSS_SOURCE_DIR}/cmake/")
INCLUDE(GetGitRevisionDescription)
GET_GIT_HEAD_REVISION(GIT_RESPEC_MODEL GIT_SHA1_MODEL)

## Build Configuration Header File
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/mobile_grid_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/mobile_grid_config.h @ONLY)
SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_SOURCE_DIR}/mobile_grid_config.h.in PROPERTIES GENERATED FALSE)
SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_BINARY_DIR}/mobile_grid_config.h PROPERTIES GENERATED TRUE)

# All source files for the model
SET(mobile_grid_lib_srcs
	#mobile_grid_main.c
	mobile_grid_map.c
	client.c
	channel.c
	coordinator.c
	aggregator.c
	selector.c
)

ADD_LIBRARY(lmobile_grid SHARED ${mobile_grid_lib_srcs})

#message(WARNING "ADDING CUDA SUBDIRECTORY")
#ADD_SUBDIRECTORY(cuda/)

ADD_EXECUTABLE(mobile_grid mobile_grid_main.c)
TARGET_LINK_LIBRARIES(mobile_grid ROSS m lmobile_grid lchannel_capacity)


# Set the executable name here: model
# You will be executing the ./mobile_grid binary
# ADD_EXECUTABLE(mobile_grid ${mobile_grid_srcs})

# TARGET_LINK_LIBRARIES(mobile_grid ROSS m)
## Alternatively, use this command with RIO
#TARGET_LINK_LIBRARIES(model ROSS RIO m)
