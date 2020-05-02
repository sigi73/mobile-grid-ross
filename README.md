# Usage

If you are creating your own model feel free to fork this repository.
As you develop, please replace "model" with appropriately descriptive names for your variables, files, and functions.

# Installation

This model can be built by ROSS by sym-linking it into the ROSS/models directory and building with `-DROSS_BUILD_MODELS=ON`

``` shell
git clone https://github.com/ROSS-org/ROSS
git clone https://github.com/sigi73/mobile-grid-ross
cd ROSS/models
ln -s ../../mobile-grid-ross ./
cd ../
mkdir build
cd build
cmake .. -DROSS_BUILD_MODELS=ON
make
./models/mobile-grid-ross/mobile_grid
```

Pass `-DCMAKE_BUILD_TYPE=Debug` for compilation with debugging

Pass `-DROSS_BUILD_DOXYGEN` to generate Doxygen files

If client side searching is desired for Doxygen, set SEARCHENGINE=YES in ROSS/docs/Doxyfile.user.in
