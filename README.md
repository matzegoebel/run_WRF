# Readme
## Usage

**Usage**: `python submit_jobs.py [OPTIONS]`
print help: `python submit_jobs.py --help`

This package helps running idealized WRF experiments. You can define a grid of namelist parameters and additional settings for your experiments in the config file `configs/config.py`. Any namelist parameter can be added to the settings. However, it has to be already defined in the default namelist file of the used WRF build directory (`WRF${wrf_dir_pre}/test/${ideal_case}/namelist.input`). Otherwise, the namelist settings cannot be overridden and an error is raised. The two variables in the path above are specified in the config file.  
The input sounding is selected in the config file. This input sounding also needs to be in the ideal case directory of the WRF build.

In addition, the script makes it easy to set the vertical and horizontal domain, start and end times of the simulations and output directories

For additional parameters in the config file, that are not namelist parameters, refer to the comments in the config file.

To initialize the simulations, run `python submit_jobs.py` with the `-i` flag and then again without this flag to start the simulations. The python script executes the bash scripts `init_wrf.job` and `run_wrf.job`, respectively. The former script copies the binaries of a specified WRF build to a new simulation directory, modifies the namelist parameters, selects the desired input sounding and executes `ideal.exe` and the latter executes `wrf.exe` in serial or parallel mode.
When run in parallel mode, the script assumes the name of the parallel build directory ends with "_mpi".

With the `-c` option you can specify an alternative config file (default is `config`).

The log output is written to `init.log/init.err` and `run.log/run.err` in the respective simulation folder for initialization and simulation mode, respectively. The subdirectory for the simulation output can be specified in the script or with the `-o` option.

If the simulation folder (in init mode) or the output files (in simulation mode) already exist, the desired action can be specified with the `-e` option: Skipping this run (`-e s`), overwriting (`-e o`) or backing up the data (`-e b`).

When run on a cluster, the `-q` flag allows submitting the jobs with SGE. Email settings for SGE can be set with the  `-m` option (set your email address in the job scripts `init_wrf.job` and `run_wrf.job` beforehand). To control which modules are loaded for cluster jobs, take a look at `init_wrf.job` and `run_wrf.job`.

 The package simplifies requesting job resources like virtual memory, runtime and number of CPUs. If the runtime of jobs is not specified in the config file, the program searches for simulations with an identical namelist file (except for some parameters irrelevant to the runtime per time step) and uses the runtime information in the respective log file. For this purpose, you can create short test simulations (with `-q` option) by setting a small value for `rt` and one repitition per configuration (`n_rep=1`) in your config file.

If a simulation aborts, simply run `submit_jobs.py -r`. This restarts the simulations from the most recent restart files. The original output is moved to a backup folder called `rst`. An unlimited number of restarts is possible. To concatenate the files from the original and the restarted runs (with overlap removed), use the script `concat_restart.py` after all runs are finished.

The option `-t` allows checking the functioning of the python script without submitting the jobs.

The `-d` option leads to "_debug" being appended to the build directory name. This is for convenience, when you want to debug `ideal.exe` or `wrf.exe` with `gdb`, subsequently. The respective WRF build, must be configured with `-d` or `-D`. 

If you want to run several simulations on the same cluster node, you can use the `-p` option. This gathers jobs until the specified pool size is reached and pins them to specific slots on the requested node. If you do not want to share the node with other users, you can fill up the whole node by specifying a pool size as large as the available slots on a node.

## Requirements
The package is written for a Linux environment. For Windows, it may have to be adjusted.
In addition to the standard library, the Python packages `numpy` and `pandas` are required. To concatenate the output of restarted and original runs, you also need the program `ncks` from the [NCO](http://nco.sourceforge.net/) package and the Python packages `netcdf4-python` and `wrf-python`.

## Testing
The folder `tests` contains scripts and data for testing the code after changing it.
Type `pytest` in the command line to run the test suite and see if everything still works as expected.
By default, for this to work, the ideal case `em_les` must be compiled in serial and parallel mode and the build directories `WRF_test` and `WRF_test_mpi` must reside in `$wrf_builds/test`.

## Known problems
#TODO not yet working for nested runs

## Getting involved
Feel free to report [issues](https://git.uibk.ac.at/csat8800/run_wrf/issues) on Gitlab.
You are also invited to improve the code and implement new features yourself. Your changes can be integrated with a [merge request](https://git.uibk.ac.at/csat8800/run_wrf/merge_requests).
Especially, further tests for the automated testing suite are appreciated.

#TODO add to PYTHONPATH
#TODO config.py is copy of config_test.py
#TODO environment variables wrf_runs,...
