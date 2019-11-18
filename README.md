# Readme
## Usage

**Usage**: `python submit_jobs.py [OPTIONS]`

This package helps running idealized WRF experiments. In the file `submit_jobs.py` you can define a grid of namelist parameters and additional settings for your experiments. Any namelist parameter can be added to the settings. These have to be already defined in the default namelist file of the used WRF build directories. Otherwise the namelist settings cannot not be overriden and a warning is raised.

To initialize the simulations (`ideal.exe`), run `python submit_jobs.py` with the `-i` flag and then again without this flag to start the simulations (`wrf.exe`). The python script executes the bash scripts `init_wrf.job` and `run_wrf.job`, respectively. The former script copies the binaries of a specified WRF build to a new simulation directory, modifies the namelist parameters, selects the desired input sounding and the latter runs the binaries in serial or parallel mode.
When run in parallel mode, the script assumes the name of the parallel build directory ends with "_mpi".


The log output is written to `init.log/init.err` and `run.log/run.err` in the respective simulation folder for initialization and simulation mode, respectively. The subdirectory for the simulation output can be specified in the script or with the `-o` option.

When run on a cluster the `-q` flag allows submitting the jobs with SGE. Email settings for SGE can be set with the  `-m` option (set your email address in the job scripts `init_wrf.job` and `run_wrf.job` beforehand).
 
In addition, the script makes it easy to set the vertical and horizontal domain and request job resources like virtual memory, runtime and number of CPUs. To control which modules are loaded for cluster jobs, take a look at `init_wrf.job` and `run_wrf.job`.

If a simulation aborts, simply run `submit_jobs.py -r`. This restarts the simulations from the most recent restart files. The original output is moved to a backup folder. To concatenate the files from the original and the restarted runs (with overlap removed), use the script `concat_restart.py` after all runs are finished.

The option `-t` allows checking the functioning of the python script without submitting the jobs.

The `-d` option leads to "_debug" being appended to the build directory name. This is for convenience, when you want to debug `ideal.exe` or `wrf.exe` with `gdb`, subsequently.The respective WRF build, must be configured with `-d` or `-D`. 

If you want to run a number of simulations on the same cluster node, you can use the `-p` option. This gathers jobs until the specified pool size is reached and pins them to specific slots on the requested node. If you do not want to share the node with other users, you can fill up the whole node by specifying a pool size as large as the available slots on a node.


## Requirements
The package is written for a Linux environment. For Windows it may have to be adjusted.
In addition to the standard library, the Python packages `numpy` and `pandas` are required. To concatenate the output of restarted and original runs, you also need the program `ncks` from the [NCO](http://nco.sourceforge.net/) package and the Python packages `netcdf4-python` and `wrf-python`.

