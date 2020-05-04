# Usage

If you are creating your own model feel free to fork this repository.
As you develop, please replace "model" with appropriately descriptive names for your variables, files, and functions.

# Installation (Locally)

This model can be built by ROSS by sym-linking it into the ROSS/models directory and building with `-DROSS_BUILD_MODELS=ON`

``` shell
git clone https://github.com/ROSS-org/ROSS
git clone https://github.com/sigi73/mobile-grid-ross
cd ROSS/models
ln -s ../../mobile-grid-ross ./
cd ../
mkdir build
cd build
cmake .. -DROSS_BUILD_MODELS=ON -DMOBILE_GRID_USE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug
make
./models/mobile-grid-ross/mobile_grid
```
Pass `-DCMAKE_BUILD_TYPE=Debug` for compilation with debugging

Pass `-DROSS_BUILD_DOXYGEN` to generate Doxygen files

Remove the -DMOBILE_GRID_USE_CUDA=ON option to not use the cuda implementation of the wireless channel


If client side searching is desired for Doxygen, set SEARCHENGINE=YES in ROSS/docs/Doxyfile.user.in



# Building on AiMOS
Copy ROSS and mobile-grid-ross to AiMOS to a folder $WORKING_DIRECTORY. Folder structure should be:
```shell
USERNAME@dcsfen01 WORKING_DIRECTORY]$ ls
mobile-grid-ross  ROSS
```
Make sure you are on dcsfen01 or dcsfen02

Then run the following commands
```shell
cd ROSS/models
ln -s ../../mobile-grid-ross ./
cd ../
mkdir build_cuda
cd build_cuda
module load spectrum-mpi gcc/7.4.0/1 cuda cmake
export CC=mpicc
cmake .. -DROSS_BUILD_MODELS=ON -DMOBILE_GRID_USE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug
make
```

# Running on AiMOS
In `mobile-grid-ross/GenData/slurmSpectrumCUDA.sh`, edit the path to the executable. It should be `$WORKING_DIRECTORY/ROSS/build_cuda/models/mobile-grid-ross/mobile_grid`

Run the script `mobile-grid-ross/GenData/sample_run.sh`
Run `./sample_run.sh 1` for an example running on 1 node
Run `./sample_run.sh 2` for an example running on 2 nodes
Ouput will be placed in `mobile-grid-ross/GenData/sample_out/`



## To collect and analyze data:
``` shell
./models/mobile-grid-ross/mobile_grid --event-trace=1
```
This will generate a stats-output directory which will contain two files: `ross-stats-evtrace.bin` and `run_statistics.csv`

Then in the mobile-grid-ross/ResultsParsing folder do
``` shell
python parse_events.py ../../ROSS/build/stats-output/ross-stats-evtrace.bin ../../ROSS/build/stats-output/run_statistics.csv
```
## To run on multiple nodes (on your computer, not AiMOS)
``` shell
mpirun -n # ./models/mobile-grid-rss/mobile_grid --synch=2
```
Where # is the number of processes. --synch=2 uses the Conservative parallel mode

# Debugging

Compile/Test TLM code with `nvcc -D TEST_TLM cuda/*.cu`


# USEFUL COMMANDS
```shell
module load spectrum-mpi gcc/7.4.0/1 cmake cuda
cmake .. -DROSS_BUILD_MODELS=ON -DMOBILE_GRID_USE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug
rsync -P -a --exclude .git --exclude build --exclude build_cuda --exclude srw --exclude suspend-test ./{mobile-grid-ross,ROSS} USERNAME@lp01.ccni.rpi.edu:~/barn/Final
```
