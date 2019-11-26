# Readme
## Usage

**Usage**: `python submit_jobs.py [OPTIONS]`
print help: `python submit_jobs.py --help`

This package helps running idealized WRF experiments. You can define a grid of namelist parameters and additional settings for your experiments in the config file `configs/config.py`. Any namelist parameter can be added to the settings. However, it has to be already defined in the default namelist file of the used WRF build directory (`WRF${wrf_dir_pre}/test/${ideal_case}/namelist.input`). Otherwise the namelist settings cannot not be overriden and a warning is raised. The two variables in the path above are specified in the config file.  
The input sounding is selected in the config file. This input sounding also needs to be in the ideal case directory of the WRF build.
In addition, the script makes it easy to set the vertical and horizontal domain and request job resources like virtual memory, runtime and number of CPUs. If the runtime of jobs is not specified in the config file, the program searches for simulations with an identical namelist file (except for some parameters irrelevant to the runtime per time step) and uses the runtime information in the respective log file. For this purpose, you can use the `-T` option to create short test simulations (with runtime `rt_check` defined in the config file). 

For additional parameters in the config file, that are not namelist parameters, refer to the comments in the config file.

To initialize the simulations (`ideal.exe`), run `python submit_jobs.py` with the `-i` flag and then again without this flag to start the simulations (`wrf.exe`). The python script executes the bash scripts `init_wrf.job` and `run_wrf.job`, respectively. The former script copies the binaries of a specified WRF build to a new simulation directory, modifies the namelist parameters, selects the desired input sounding and the latter runs the binaries in serial or parallel mode.
When run in parallel mode, the script assumes the name of the parallel build directory ends with "_mpi".

With the `-c` option you can specify an alternative config file (default is `config`).

The log output is written to `init.log/init.err` and `run.log/run.err` in the respective simulation folder for initialization and simulation mode, respectively. The subdirectory for the simulation output can be specified in the script or with the `-o` option.

When run on a cluster, the `-q` flag allows submitting the jobs with SGE. Email settings for SGE can be set with the  `-m` option (set your email address in the job scripts `init_wrf.job` and `run_wrf.job` beforehand). To control which modules are loaded for cluster jobs, take a look at `init_wrf.job` and `run_wrf.job`.

If a simulation aborts, simply run `submit_jobs.py -r`. This restarts the simulations from the most recent restart files. The original output is moved to a backup folder called `rst`. Unlimited number of restarts are possible. To concatenate the files from the original and the restarted runs (with overlap removed), use the script `concat_restart.py` after all runs are finished.

The option `-t` allows checking the functioning of the python script without submitting the jobs.

The `-d` option leads to "_debug" being appended to the build directory name. This is for convenience, when you want to debug `ideal.exe` or `wrf.exe` with `gdb`, subsequently.The respective WRF build, must be configured with `-d` or `-D`. 

If you want to run a number of simulations on the same cluster node, you can use the `-p` option. This gathers jobs until the specified pool size is reached and pins them to specific slots on the requested node. If you do not want to share the node with other users, you can fill up the whole node by specifying a pool size as large as the available slots on a node.


## Requirements
The package is written for a Linux environment. For Windows it may have to be adjusted.
In addition to the standard library, the Python packages `numpy` and `pandas` are required. To concatenate the output of restarted and original runs, you also need the program `ncks` from the [NCO](http://nco.sourceforge.net/) package and the Python packages `netcdf4-python` and `wrf-python`.

