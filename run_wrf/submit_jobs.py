#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Automatically initialize and run idealized experiments in WRF
on a single computer or cluster.

@author: Matthias Göbel
"""
import numpy as np
import math
import os
import datetime
import argparse
import glob
import importlib
import inspect
from copy import deepcopy
from pathlib import Path as fopen
import sys

# %%


def submit_jobs(config_file="config", init=False, outpath=None, exist="s",
                debug=False, use_job_scheduler=False, check_args=False,
                pool_jobs=False, mail="ea", wait=False, no_namelist_check=False,
                test_run=False, verbose=False, param_combs=None):
    """
    Initialize and run idealized WRF experiments.
    Refer to README.md for more information.

    Parameters
    ----------
    config_file : str, optional
        Name of config file in configs folder. The default is "config".
    init : bool, optional
        Initialize simulations.
    outpath : str, optional
        Directory for WRF output. Default defined in script. Only effective during initialization.
    exist : str, optional
        What to do if output already exists: Skip run ('s'), overwrite ('o'),
        restart ('r') or backup files ('b'). Default is 's'.
    debug : bool, optional
        Run wrf in debugging mode. Just adds '_debug' to the build directory.
    use_job_scheduler : bool, optional
        Use job scheduler to submit jobs
    check_args : bool, optional
        Only test python script (no jobs sumitted)
    pool_jobs : bool, optional
        Gather jobs before submitting with a job scheduler.
        Needed if different jobs shall be run on the some node
        (potentially filling up the whole node)
    mail : str, optional
        If using a job scheduler, defines when mail is sent.
        Either 'n' for no mails, or a combination of 'b' (beginning of job),
        'e' (end), 'a' (abort)', 's' (suspended). Default: 'ea'
    wait : bool, optional
        Wait until job is finished before submitting the next.
    no_namelist_check : bool, optional
        Do not perform sanity check of namelist parameters.
    test_run : bool, optional
        Do short test runs on cluster to find out required runtime and virtual memory
    verbose : bool, optional
        Verbose mode
    param_combs : list of dicts or pandas DataFrame, optional
        DataFrame with settings for all configurations.
    Returns
    -------
    param_combs : pandas DataFrame
        DataFrame with settings for all configurations.

    """
    from run_wrf import misc_tools

    if (not init) and (outpath is not None):
        print("WARNING: option -o ignored when not in initialization mode!\n")
    if wait and use_job_scheduler:
        raise ValueError("Waiting for batch jobs is not yet implemented")
    if init and (exist == "r"):
        raise ValueError("For restart runs no initialization is needed!")

    if config_file[-3:] == ".py":
        config_file = config_file[:-3]
    try:
        conf = importlib.import_module("run_wrf.configs.{}".format(config_file))
    except ModuleNotFoundError:
        sys.path.append(os.getcwd())
        conf = importlib.import_module(config_file)

    # change to code path
    fpath = os.path.realpath(__file__)
    os.chdir(fpath[:fpath.index("submit_jobs.py")])

    if param_combs is None:
        if "param_combs" in dir(conf):
            param_combs = conf.param_combs
        else:
            args = []
            for k in ["param_grid", "params", "param_names"]:
                if k in dir(conf):
                    args.append(eval("conf.{}".format(k)))
                else:
                    args.append(None)

            param_combs = misc_tools.grid_combinations(*args, runID=conf.runID)

    param_combs = deepcopy(param_combs)

    if test_run:
        print("Do short test runs on cluster to find out required runtime and virtual memory\n\n")

    if outpath is None:
        base_outpath = conf.outpath
    else:
        base_outpath = outpath

    if init:
        job_name = "init_"
    else:
        job_name = "run_"
    job_name += conf.runID
    # temporary log output for job scheduler
    if use_job_scheduler:
        batch_log_dir = conf.run_path + "/logs/"
        os.makedirs(batch_log_dir, exist_ok=True)
        job_scheduler = conf.job_scheduler.lower()
        if job_scheduler not in ["slurm", "sge"]:
            raise ValueError("Job scheduler {} not implemented. "
                             "Use SGE or SLURM".format(job_scheduler))
        if job_scheduler == "slurm":
            # assuming no oversubscription is allowed, pooling is necessary
            if conf.force_pool:
                pool_jobs = True
            mail_slurm = []
            for s, r in zip(["n", "b", "e", "a"], ["NONE", "BEGIN", "END", "FAIL"]):
                if s in mail:
                    mail_slurm.append(r)
            mail = ",".join(mail_slurm)
        if (conf.mail_address is None) or (conf.mail_address == ""):
            raise ValueError("For jobs using {}, provide valid mail address "
                             "in config file".format(job_scheduler))

        if job_scheduler == "slurm":
            job_id = "_%j"
        else:
            job_id = "_$JOB_ID"
    else:
        job_scheduler = None
        conf.request_vmem = False

    # if test_run and (job_scheduler == "sge"):
    #     #do test run on one node by using openmpi-xperhost to ensure correct vmem logging
    #     conf.reduce_pool = True

    IDs = []
    rtr = []
    vmem = []
    nslots = []
    nxny = []

    if init:
        print("Initialize WRF simulations")
    else:
        print("Run WRF simulations")

    print("Configs:")
    if "core_param" in param_combs.index:
        core_params = param_combs.loc["core_param"]
        # delete core_param line and composite_idx lines and columns
        composite_idx = param_combs.iloc[-1]
        param_combs = param_combs.loc[:, ~composite_idx.astype(bool)]
        param_combs = param_combs.iloc[:-2]
        # print only core parameters
        print(param_combs.loc[:, core_params])

    else:
        print(param_combs.index.values)
    print("-" * 40)
    for i, (cname, param_comb) in enumerate(param_combs.iterrows()):
        IDi = param_comb["fname"]
        args = deepcopy(param_comb.dropna().to_dict())
        del args["fname"]

        # create output ID for current configuration
        run_dir = "{}/WRF_{}".format(conf.run_path, IDi)

        print("\n\nConfig:  " + IDi)
        print(cname)
        print("\n")

        if ("dy" not in args) and ("dx" in args):
            args["dy"] = args["dx"]

        # start and end times
        date_format = '%Y-%m-%d_%H:%M:%S'
        start_time_dt = datetime.datetime.strptime(args["start_time"], date_format)
        end_time_dt = datetime.datetime.strptime(args["end_time"], date_format)
        start_d, start_t = args["start_time"].split("_")
        start_d = start_d.split("-")
        start_t = start_t.split(":")
        end_d, end_t = args["end_time"].split("_")
        end_d = end_d.split("-")
        end_t = end_t.split(":")

        run_hours = (end_time_dt - start_time_dt).total_seconds() / 3600
        if run_hours <= 0:
            raise ValueError("Selected end time {} smaller or equal start time {}!".format(
                args["end_time"], args["start_time"]))

        for di, n in zip(start_d + start_t, ["year", "month", "day", "hour", "minute", "second"]):
            args["start_" + n] = di
        for di, n in zip(end_d + end_t, ["year", "month", "day", "hour", "minute", "second"]):
            args["end_" + n] = di
        # use end time not run_*
        args["run_hours"] = 0
        args["run_minutes"] = 0
        args["run_seconds"] = 0

        # horizontal domain
        if "use_min_gridpoints" in dir(conf):
            gp = conf.use_min_gridpoints
        else:
            gp = False
        if "use_min_gridpoints" in dir(conf):
            fm = conf.force_domain_multiple
        else:
            fm = False

        if ("lx" in conf.params) and ("dx" in conf.params):
            if (not gp) or (gp == "y"):
                args["e_we"] = math.ceil(args["lx"] / args["dx"]) + 1
            else:
                args["e_we"] = max(math.ceil(args["lx"] / args["dx"]),
                                   args["min_gridpoints_x"] - 1) + 1
                if (fm is True) or (fm == "x"):
                    lxr = (args["e_we"] - 1) * args["dx"] / args["lx"]
                    if lxr != int(lxr):
                        raise Exception("Domain size must be multiple of lx")

        if ("ly" in conf.params) and ("dy" in conf.params):
            if (not gp) or (gp == "x"):
                args["e_sn"] = math.ceil(args["ly"] / args["dy"]) + 1
            else:
                args["e_sn"] = max(math.ceil(args["ly"] / args["dy"]),
                                   args["min_gridpoints_y"] - 1) + 1
                if (fm is True) or (fm == "y"):
                    lyr = (args["e_sn"] - 1) * args["dy"] / args["ly"]
                    if lyr != int(lyr):
                        raise Exception("Domain size must be multiple of ly")

        # slots
        nx = misc_tools.find_nproc(args["e_we"] - 1, min_n_per_proc=conf.min_nx_per_proc,
                                   even_split=conf.even_split)
        ny = misc_tools.find_nproc(args["e_sn"] - 1, min_n_per_proc=conf.min_ny_per_proc,
                                   even_split=conf.even_split)

        if conf.max_nslotsx is not None:
            nx = min(conf.max_nslotsx, nx)
        if conf.max_nslotsy is not None:
            ny = min(conf.max_nslotsy, ny)

        if (nx == 1) and (ny == 1):
            nx = -1
            ny = -1
        nslotsi = nx * ny

        # determine which build to use
        slot_comm = ""
        if debug:
            wrf_dir_i = conf.debug_build
        elif (np.array([*nslots, nslotsi]) > 1).any():
            wrf_dir_i = conf.parallel_build
            if use_job_scheduler and (not pool_jobs):
                if job_scheduler == "sge":
                    slot_comm = "-pe openmpi-fillup {}".format(nslotsi)
                elif job_scheduler == "slurm":
                    slot_comm = "-N {}".format(nslotsi)
        else:
            wrf_dir_i = conf.serial_build

        vmemi = None
        if init:
            wrf_build = os.path.join(conf.build_path, wrf_dir_i)
            print("Using WRF build in: {}\n".format(wrf_build))

            args, args_str = misc_tools.prepare_init(args, conf, wrf_dir_i,
                                                     namelist_check=not no_namelist_check)
            # job scheduler queue and vmem
            if use_job_scheduler and conf.request_vmem:
                vmem_grid = conf.vmem_init_per_grid_point * args["e_we"] * args["e_sn"]
                vmem_init = max(conf.vmem_init_min, int(vmem_grid))
                if ("bigmem_limit" in dir(conf)) and (vmem_init > conf.bigmem_limit):
                    queue = conf.bigmem_queue
                else:
                    queue = conf.queue

        elif use_job_scheduler or test_run:
            queue = conf.queue
            args, skip = misc_tools.set_vmem_rt(args, run_dir, conf, run_hours, nslots=nslotsi,
                                                pool_jobs=pool_jobs, test_run=test_run,
                                                request_vmem=conf.request_vmem)
            if skip:
                continue
            if conf.request_vmem:
                vmemi = args["vmem"]

        # not needed; just for completeness of dataframe:
        args["nx"] = nx
        args["ny"] = ny
        args["nslots"] = nslotsi
        args["run_dir"] = run_dir
        for arg, val in args.items():
            if arg not in param_combs.keys():
                param_combs[arg] = None
            param_combs[arg][cname] = val

        n_rep = args.setdefault("n_rep", 1)
        for rep in range(n_rep):  # repetion loop
            vmem.append(vmemi)
            nslots.append(nslotsi)
            nxny.append([nx, ny])

            IDr = IDi + "_" + str(rep)
            run_dir_r = run_dir + "_" + str(rep)

            # create output path
            outpath = os.path.join(base_outpath, IDr, "")  # WRF output path
            if not os.path.isdir(outpath):
                os.makedirs(outpath)
            outpath_esc = outpath.replace("/", "\/")  # need to escape slashes

            if init:
                if os.path.isdir(run_dir_r):
                    if exist == "s":
                        print("Run directory already exists.")
                        if os.path.isfile(run_dir_r + "/wrfinput_d01"):
                            print("Initialization was complete.\nSkipping...")
                            continue
                        else:
                            print("However, WRF initialization was not successfully carried out."
                                  "\nRedoing initialization...")
                    elif exist == "o":
                        print("Overwriting...")
                    elif exist == "b":
                        print("Creating backup...")
                        bk_dir = "{}/bak/".format(conf.run_path)
                        run_dir_bk = "{}/WRF_{}_bak_".format(bk_dir, IDr)
                        os.makedirs(bk_dir, exist_ok=True)
                        bk_ind = 0
                        while os.path.isdir(run_dir_bk + str(bk_ind)):
                            bk_ind += 1
                        os.rename(run_dir_r, run_dir_bk + str(bk_ind))
                    else:
                        raise ValueError("Value '{}' for -e option not defined!".format(exist))

                hist_paths = r""
                for stream, (outfile, _) in args["output_streams"].items():
                    outname = r"{}{}_d<domain>_<date>".format(outpath_esc, outfile)

                    if stream == 0:
                        stream_arg = "history_outname"
                    else:
                        stream_arg = "auxhist{}_outname".format(stream)

                    hist_paths = r'''{} {} "{}"'''.format(hist_paths, stream_arg, outname)

                args_str_r = args_str + hist_paths

                iofile = ""
                if "iofields_filename" in args:
                    iofile_ = args["iofields_filename"].replace("'", "").replace('"', "")
                    if iofile_ != "NONE_SPECIFIED":
                        iofile = iofile_

                comm_args = dict(run_id=IDr, wrfv=wrf_dir_i, ideal_case=conf.ideal_case,
                                 input_sounding=args["input_sounding"], nx=nx, ny=ny,
                                 run_path=conf.run_path, build_path=conf.build_path,
                                 batch=int(use_job_scheduler), wrf_args="",
                                 cluster=int(conf.cluster), iofile=iofile,
                                 module_load=conf.module_load)
                for p, v in comm_args.items():
                    os.environ[p] = str(v)
                if use_job_scheduler:
                    os.environ["job_scheduler"] = job_scheduler
                    os.environ["wrf_args"] = args_str_r
                    rt_init = misc_tools.format_timedelta(conf.rt_init * 60)
                    qlog = batch_log_dir + job_name
                    os.environ["qlog"] = qlog
                    qout, qerr = [qlog + job_id + s for s in [".out", ".err"]]
                    batch_args = [queue, qout, qerr, rt_init, conf.mail_address, mail, job_name]
                    if job_scheduler == "sge":
                        batch_args_str = "qsub -cwd -q {} -o {} -e {} -l h_rt={} -M " \
                                         "{} -m {} -N {} -V ".format(*batch_args)
                        if "h_stack_init" in dir(conf) and conf.h_stack_init is not None:
                            batch_args_str += " -l h_stack={}M ".format(round(conf.h_stack_init))
                        if conf.request_vmem:
                            batch_args_str += " -l h_vmem={}M ".format(vmem_init)

                    elif job_scheduler == "slurm":
                        batch_args_str = "sbatch --qos={}  -p {} -o {} -e {} --time={} " \
                            "--mail-user={} --mail-type={} -J {} -N 1 -n 1 " \
                            "--export=ALL ".format(conf.qos, *batch_args)
                        if conf.request_vmem:
                            batch_args_str += " --mem-per-cpu={}M ".format(vmem_init)

                    comm = batch_args_str + " init_wrf.job"
                else:
                    comm = "bash init_wrf.job '{}' ".format(args_str_r)

                if verbose:
                    print(comm)
                if not check_args:
                    err = os.system(comm)

                    initlog = fopen(run_dir_r + "/init.log").read_text()
                    if "wrf: SUCCESS COMPLETE IDEAL INIT" in initlog:
                        print("wrf: SUCCESS COMPLETE IDEAL INIT")
                    else:
                        initerr = fopen(run_dir_r + "/init.err").read_text()
                        print("\n")
                        print(initerr)
                        raise RuntimeError("Initialization failed!")

            else:
                skip = False
                if not os.path.isfile(run_dir_r + "/wrfinput_d01"):
                    print("Run not initialized yet! Skipping...")
                    skip = True
                stream_names = [stream[0] for stream in args["output_streams"].values()]
                outfiles = []
                for outfile in stream_names:
                    outfiles.extend(glob.glob("{}/{}*".format(outpath, outfile)))
                restart = False
                if (len(outfiles) > 0) and (not skip):
                    print("Output files already exist.")
                    if exist == "s":
                        print("Skipping...")
                        skip = True

                    elif exist == "o":
                        print("Overwriting...")
                    elif exist == "b":
                        print("Creating backup...")
                        bk_dir = "{}/bak/".format(outpath)
                        os.makedirs(bk_dir, exist_ok=True)
                        for outfile in outfiles:
                            outfile_name = outfile.split("/")[-1]
                            outfile_bk = "{}/{}_bak_".format(bk_dir, outfile_name)
                            bk_ind = 0
                            while os.path.isfile(outfile_bk + str(bk_ind)):
                                bk_ind += 1
                            os.rename(outfile, outfile_bk + str(bk_ind))
                    elif exist == "r":
                        restart = True
                    else:
                        raise ValueError("Value '{}' for -e option not defined!".format(exist))

                if restart:
                    run_hours_rst, rst_opt = misc_tools.get_restart_times(run_dir_r, args["end_time"])
                    if run_hours_rst is None:
                        restart = False
                    elif run_hours_rst <= 0:
                        skip = True
                    else:
                        run_hours = run_hours_rst

                last_id = False
                if (rep == n_rep - 1) and (i == len(param_combs) - 1):
                    last_id = True
                if skip:
                    vmem = vmem[:-1]
                    nslots = nslots[:-1]
                    nxny = nxny[:-1]
                    # if last ID: do run but without current config, else: skip run
                    if (not last_id) or (len(nslots) == 0):
                        continue
                else:
                    IDs.append(IDr)
                    rtri = None
                    if use_job_scheduler or test_run:
                        # runtime in seconds
                        rtri = args["rt_per_timestep"] * run_hours * args["dt_f"] * 3600
                        rtr.append(rtri)

                if (not pool_jobs) or (sum(nslots) >= conf.pool_size) or last_id:
                    # submit current pool of jobs
                    print("")
                    resched_i = False
                    # if pool is already too large: cut out last job,
                    # which is rescheduled afterwards
                    if pool_jobs and (sum(nslots) > conf.pool_size):
                        if len(nslots) > 1:
                            nslots = nslots[:-1]
                            nxny = nxny[:-1]
                            vmem = vmem[:-1]
                            rtr = rtr[:-1]
                            IDs = IDs[:-1]
                            resched_i = True
                        elif job_scheduler == "sge":
                            raise ValueError("Pool size ({}) smaller than number of slots of "
                                             "current job ({})!".format(conf.pool_size, nslots[0]))

                    iterate = True
                    while iterate:
                        iterate = False
                        print("Submit IDs: {}".format(IDs))
                        print("with total cores: {}".format(sum(nslots)))
                        if pool_jobs and use_job_scheduler:
                            if job_scheduler == "sge":
                                nperhost = conf.pool_size
                                if conf.reduce_pool:
                                    # select smallest available nperhost that is greater or
                                    # equal to the number of requested slots
                                    pes = os.popen("qconf -spl").read().split("\n")
                                    nperhost_avail = np.array([int(pe[8:pe.index("per")])
                                                               for pe in pes if "perhost" in pe])
                                    nperhost = nperhost_avail[nperhost_avail >= sum(nslots)].min()
                                slot_comm = "-pe openmpi-{0}perhost {0}".format(nperhost)
                            elif job_scheduler == "slurm":
                                nodes = math.ceil(sum(nslots) / conf.pool_size)
                                if nodes == 1:
                                    ntasks = sum(nslots)
                                else:
                                    ntasks = conf.pool_size
                                slot_comm = "--ntasks-per-node={} -N {}".format(ntasks, nodes)

                        jobs = " ".join(IDs)
                        nslots_str = " ".join([str(ns) for ns in nslots])
                        nx_str = " ".join([str(ns[0]) for ns in nxny])
                        ny_str = " ".join([str(ns[1]) for ns in nxny])
                        timestamp = datetime.datetime.now().isoformat()[:19]
                        comm_args = dict(nslots=nslots_str, nx=nx_str, ny=ny_str, jobs=jobs,
                                         pool_jobs=int(pool_jobs), run_path=conf.run_path,
                                         batch=int(use_job_scheduler), cluster=int(conf.cluster),
                                         restart=int(restart), outpath=outpath,
                                         module_load=conf.module_load, timestamp=timestamp)
                        for p, v in comm_args.items():
                            os.environ[p] = str(v)

                        if use_job_scheduler or test_run:
                            rtr_max = max(rtr)

                        if use_job_scheduler:
                            os.environ["job_scheduler"] = job_scheduler
                            send_rt_signal = conf.send_rt_signal

                            if conf.request_vmem:
                                vmemp = int(sum(vmem) / sum(nslots))
                                if ("bigmem_limit" in dir(conf)) and (vmemp > conf.bigmem_limit):
                                    queue = conf.bigmem_queue

                            rtp = misc_tools.format_timedelta(rtr_max)
                            if rtr_max < send_rt_signal:
                                raise ValueError("Requested runtime is smaller then the time "
                                                 "when the runtime limit signal is sent!")

                            qlog = batch_log_dir + job_name
                            os.environ["qlog"] = qlog
                            qout, qerr = [qlog + job_id + s for s in [".out", ".err"]]
                            batch_args = [conf.queue, qout, qerr, rtp,
                                          slot_comm, conf.mail_address, mail, job_name]
                            os.environ["rtlimit"] = str(int(rtr_max - send_rt_signal))

                            if job_scheduler == "sge":
                                batch_args_str = "qsub -cwd -q {} -o {} -e {} -l h_rt={}  {} " \
                                                 " -M {} -m {} -N {} -V ".format(*batch_args)
                                if "h_stack" in dir(conf) and conf.h_stack is not None:
                                    batch_args_str += " -l h_stack={}M ".format(round(conf.h_stack))
                                if conf.request_vmem:
                                    batch_args_str += " -l h_vmem={}M ".format(vmemp)

                            elif job_scheduler == "slurm":
                                batch_args_str = "sbatch -p {} -o {} -e {} --time={} {} " \
                                                 "--mail-user={} --mail-type={} -J {} " \
                                                 "--export=ALL ".format(*batch_args)
                                if ("qos" in dir(conf)) and (conf.qos is not None):
                                    batch_args_str += " --qos={} ".format(conf.qos)
                                if conf.request_vmem:
                                    batch_args_str += " --mem-per-cpu={}M ".format(vmemp)

                            comm = batch_args_str + " run_wrf.job"

                        else:
                            if test_run:
                                os.environ["rtlimit"] = str(int(rtr_max))
                            else:
                                os.environ["rtlimit"] = ""

                            comm = "bash run_wrf.job"
                            if not wait:
                                comm += " &"

                        if verbose:
                            print(comm)
                        if not check_args:
                            if restart:
                                misc_tools.prepare_restart(run_dir_r, rst_opt)

                            err = os.system(comm)
                            if wait:
                                if restart:
                                    log_lines = 15
                                else:
                                    log_lines = 10

                                for ID in IDs:
                                    print(ID)
                                    run_dir_i = "{}/WRF_{}/".format(conf.run_path, ID)
                                    print(os.popen("tail -n {} {}/run_{}.log".format(log_lines,
                                                                                     run_dir_i, timestamp)).read())
                                    print(fopen(run_dir_i + "run_{}.err".format(timestamp)).read_text())
                            # if err != 0:
                            #     raise RuntimeError("WRF run failed!")

                        if resched_i:
                            IDs = [IDr]
                            rtr = [rtri]
                            vmem = [vmemi]
                            nslots = [nslotsi]
                            nxny = [[nx, ny]]

                        # run residual jobs that did not fit in the last job pool
                            if last_id:
                                iterate = True
                                last_id = False
                        else:
                            IDs = []
                            rtr = []
                            vmem = []
                            nslots = []
                            nxny = []

    return param_combs


# %%
if __name__ == "__main__":

    # define argument parser

    sig = inspect.signature(submit_jobs).parameters

    # default function values
    defaults = {k: d.default for k, d in sig.items()}
    del defaults["param_combs"]
    doc = submit_jobs.__doc__
    # get description from doc string
    desc = {}
    for k in defaults.keys():
        desc_k = doc[doc.index(" " + k + " "):]
        desc[k] = desc_k.split("\n")[1].strip()

    intro = doc[:doc.index("Parameters\n")]

    # command line arguments (equivalent to function arguments above)
    # short form, long form, action
    parse_params = {"config_file":       ("-c", "--config", "store"),
                    "init":              ("-i", "--init", "store_true"),
                    "outpath":           ("-o", "--outpath", "store"),
                    "exist":             ("-e", "--exist", "store"),
                    "debug":             ("-d", "--debug", "store_true"),
                    "use_job_scheduler": ("-j", "--use_job_sched", "store_true"),
                    "check_args":        ("-t", "--test", "store_true"),
                    "pool_jobs":         ("-p", "--pool", "store_true"),
                    "mail":              ("-m", "--mail", "store"),
                    "wait":              ("-w", "--wait", "store_true"),
                    "no_namelist_check": ("-n", "--no_namelist_check", "store_true"),
                    "test_run":          ("-T", "--test_run", "store_true"),
                    "verbose":           ("-v", "--verbose", "store_true")
                    }

    assert sorted(parse_params.keys()) == sorted(desc.keys())
    add_args = {"exist": {"choices": ["s", "o", "r", "b"]}}

    parser = argparse.ArgumentParser(description=intro)
    for p, default in defaults.items():
        if p in add_args:
            add_args_p = add_args[p]
        else:
            add_args_p = {}
        parser.add_argument(*parse_params[p][:-1], action=parse_params[p][-1],
                            dest=p, default=default, help=desc[p], **add_args_p)

    parser.format_help()
    options = parser.parse_args()

    param_combs = submit_jobs(**options.__dict__)
