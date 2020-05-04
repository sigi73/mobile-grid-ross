# Mobile Grid

ROSS (Rensselaer’s Optimistic Simulation System) model to simulate the use of mobile devices to perform productive computation work when otherwise idle.

ROSS: https://github.com/ROSS-org/ROSS

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
In `mobile-grid-ross/GenData/slurmSpectrumCUDA.sh`, and `mobile-grid-ross/GenData/slurmSpectrumCPU.sh`, edit the path to the executable. It should be `$WORKING_DIRECTORY/ROSS/build_cuda/models/mobile-grid-ross/mobile_grid`

Use the full path to the file. This is necessary to ensure that slurm properly launches the program.

## Run Single Job
Use the script `mobile-grid-ross/GenData/sample_run.sh`
Run `./sample_run.sh 1` for an example running on 1 node
Run `./sample_run.sh 2` for an example running on 2 nodes
Ouput will be placed in `mobile-grid-ross/GenData/sample_out/`

## Run batched job
Run the following
```shell
cd mobile-grid-ross/GenData
python gen_data.py
chmod +x gen_execute.sh
./gen_execute.sh
```
The python script `gen_data.py` generates the script `gen_execute.sh`. This runs the full set of tests used to collected data for the associated paper. NOTE: This runs > 50 jobs.

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

# Debugging CUDA TLM Code

Compile/Test TLM code with `nvcc -D TEST_TLM cuda/*.cu`


# USEFUL COMMANDS
```shell
module load spectrum-mpi gcc/7.4.0/1 cmake cuda
cmake .. -DROSS_BUILD_MODELS=ON -DMOBILE_GRID_USE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug
rsync -P -a --exclude .git --exclude build --exclude build_cuda --exclude srw --exclude suspend-test ./{mobile-grid-ross,ROSS} USERNAME@lp01.ccni.rpi.edu:~/barn/Final
```
