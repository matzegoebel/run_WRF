# Readme
## Usage

**Usage**: `submit_jobs.py [OPTIONS]`
print help: `submit_jobs.py --help`

This package helps running idealized WRF experiments. You can define a grid of namelist parameters and additional settings for your experiments in the config file `configs/config.py`. Any namelist parameter can be added to the settings. However, it has to be already defined in the default namelist file of the used WRF build directory (`WRF${wrf_dir_pre}/test/${ideal_case}/namelist.input`). Otherwise, the namelist settings cannot be overridden and an error is raised. The two variables in the path above are specified in the config file.
The input sounding is selected in the config file. This input sounding also needs to be in the ideal case directory of the WRF build.
In addition, the script makes it easy to set the vertical and horizontal domain, start and end times of the simulations and output directories
For additional parameters in the config file, that are not namelist parameters, refer to the comments in the config file.

To initialize the simulations, run `submit_jobs.py` with the `-i` flag and then again without this flag to start the simulations. The python script executes the bash scripts `init_wrf.job` and `run_wrf.job`, respectively. The former script copies the binaries of a specified WRF build to a new simulation directory, modifies the namelist parameters, selects the desired input sounding and executes `ideal.exe` and the latter executes `wrf.exe` in serial or parallel mode.
When run in parallel mode, the script assumes the name of the parallel build directory ends with "_mpi".

With the `-v` option (verbose mode) the arguments of the call to the bash scripts are displayed. In this way you can check if the namelist parameters that will be set for each configuration are as desired.

With the `-c` option you can specify an alternative config file (default is `config`) located in (a subdirectory of) the folder `configs`. For instance, the config file `test_sfclay.py` is located in `configs/example/` and can be accessed with `submit_jobs.py -i -c example.config_sfclay`. This example config file also shows how to use the parameters defined in another config file, such that only the parameters which are different in the new file have to be specified.

When initializing the simulations, the namelist settings are checked for sanity and consistency based on best practice guidelines for WRF. Warnings and---for severe problems---errors are raised. To ignore the errors and proceed with initializing the simulations, use the `-n` option.

The log output is written to `init.log/init.err` and `run_$TIME.log/run_$TIME.err` in the respective simulation folder for initialization and simulation mode, respectively. The subdirectory for the simulation output can be specified in the script or with the `-o` option. Currently, the default config file uses the environment variables `$wrf_builds`, `$wrf_runs` and `$wrf_res` as base directories for WRF build, run and output directories, respectively.

If the simulation folder (in init mode) or the output files (in simulation mode) already exist, the desired action can be specified with the `-e` option: Skipping this run (`-e s`), overwriting (`-e o`) or backing up the data (`-e b`).

When run on a cluster, the `-j` flag allows submitting the jobs with a job scheduler, `SGE` or `SLURM`. The job scheduler and other cluster specific settings such as required modules and queues can be set in the config file for each cluster in use. In the default config file settings for the LEO cluster of the University of Innsbruck and the Vienna Scientific Cluster (*VSC*) are included. Email settings for the job scheduler can be set with the `-m` option.

The package simplifies requesting job resources like virtual memory, runtime and number of CPUs. If the runtime or virtual memory of jobs is not specified in the config file, the program searches for simulations with an identical namelist file (except for some parameters irrelevant to the runtime per time step) and uses the runtime and virtual memory information in the respective log file. For this purpose, you can create short test simulations with the `-T` (and `-j`) option. This sets the number of repetitions to 1, the runtime to the value of `rt_test` and the virtual memory to `vmem_test` as defined in the config file. On the *VSC* cluster only one job per node is allowed. The specification of virtual memory is thus not necessary and omitted by default.

If not using a job scheduler, the simulations are started simultaneously and run in the background, by default. The `-w` option allows waiting for a simulation (or pool of simulations, see below) to finish, before submitting the next.

If a simulation aborts or you want to continue it to a later end time (as specified in the config file), simply run `submit_jobs.py -r`. This restarts the simulations from the most recent restart files. The original output is moved to a backup folder and concatenated with the output of the restarted run (with overlap removed) after the restarted run is finished. An unlimited number of restarts is possible.

The option `-t` allows checking the functioning of the python script without submitting the jobs.

The `-d` option leads to "_debug" being appended to the build directory name. This is for convenience, when you want to debug `ideal.exe` or `wrf.exe` with `gdb`, subsequently. The respective WRF build, must be configured with `-d` or `-D`.

If you want to run several simulations on the same cluster node, you can use the `-p` option. This gathers jobs until the specified pool size is reached and runs them simultaneously in one batch job. If you do not want to share the node with other users, you can fill up the whole node by specifying a pool size as large as the available slots on a node. On the *VSC* cluster only one job per node is allowed. Therefore job pooling is important to make best use of the available resources and thus switched on by default.
The pooling option can also be used without job scheduler. Combined with the `-w` option, you can ensure that only `pool_size` cores are used simultaneously.

## Requirements
The package is written for a Linux environment. For Windows, it may have to be adjusted.
In addition to the standard library, the Python packages `numpy`, `pandas`, and `xarray` are required. To concatenate the output of restarted and original runs, you also need the program `ncks` from the [NCO](http://nco.sourceforge.net/) package.
The package was tested with Python 3.6.

## Installation
Run `pip install -e .` in the root directory.

## Testing
The folder `tests` contains scripts and data for testing the code after changing it.
Type `pytest` in the command line to run the test suite and see if everything still works as expected.
By default, for this to work, the ideal case `em_les` must be compiled in serial and parallel mode and the build directories `WRF_test` and `WRF_test_mpi` must reside in `$wrf_builds/test`.

## Known problems
Currently, the script (especially the modification of the namelist files) does not work for nested runs, as it is meant to be used for idealized simulations, only.

## Getting involved
Feel free to report [issues](https://github.com/matzegoebel/run_wrf/issues) on Github.
You are also invited to improve the code and implement new features yourself. Your changes can be integrated with a [pull request](https://github.com/matzegoebel/run_wrf/pulls).
Especially, further tests for the automated testing suite are appreciated.
