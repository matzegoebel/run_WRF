#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:08:01 2019

Automatically initialize and run idealized experiments in WRF on a single computer or cluster.

@author: c7071088
"""
import numpy as np
import pandas as pd
import math
import os
import datetime
import argparse
import glob
import misc_tools
import importlib
import inspect
from copy import deepcopy
from pathlib import Path as fopen

#%%
def submit_jobs(config_file="config", init=False, restart=False, outdir=None, exist="s", debug=False, use_job_scheduler=False,
           check_args=False, pool_jobs=False, mail="ea", wait=False, no_namelist_check=False, test_run=False, verbose=False):
    """
    Submit idealized WRF experiments. Refer to README.md for more information.

    Parameters
    ----------
    config_file : str, optional
        Name of config file in configs folder. The default is "config".
    init : bool, optional
        Initialize simulations.
    restart : bool, optional
        Restart simulations.
    outdir : str, optional
        Subdirectory for WRF output. Default defined in script. Only effective during initialization.
    exist : str, optional
        What to do if output already exists: Skip run ('s'), overwrite ('o') or backup files ('b'). Default is 's'.
    debug : bool, optional
        Run wrf in debugging mode. Just adds '_debug' to the build directory.
    use_job_scheduler : bool, optional
        Use job scheduler to submit jobs
    check_args : bool, optional
        Only test python script (no jobs sumitted)
    pool_jobs : bool, optional
        Gather jobs before submitting with a job scheduler. Needed if different jobs shall be run on the some node (potentially filling up the whole node)
    mail : str, optional
        If using a job scheduler, defines when mail is sent. Either 'n' for no mails, or a combination of 'b' (beginning of job), 'e' (end), 'a' (abort)', 's' (suspended). Default: 'ea'
    wait : bool, optional
        Wait until job is finished before submitting the next.
    no_namelist_check : bool, optional
        Do not perform sanity check of namelist parameters.
    test_run : bool, optional
        Do short test runs on cluster to find out required runtime and virtual memory
    verbose : bool, optional
        Verbose mode

    Returns
    -------
    combs : pandas DataFrame
        DataFrame with settings for all submitted configurations.

    """

    if (not init) and (outdir is not None):
        print("WARNING: option -o ignored when not in initialization mode!\n")
    if wait and use_job_scheduler:
        raise ValueError("Waiting for batch jobs is not yet implemented")
    if test_run and (not use_job_scheduler):
        raise ValueError("Test runs without job scheduler do not make sense!")
    if test_run and pool_jobs:
        raise ValueError("Do not use pooling for test runs!")
    if init and restart:
        raise ValueError("For restart runs no initialization is needed!")
    if len(glob.glob(os.getcwd() + "/submit_jobs.py")) == 0:
        raise RuntimeError("Script must be started from within its directory!")

    if config_file[-3:] == ".py":
        config_file = config_file[:-3]
    conf = importlib.import_module("configs.{}".format(config_file))
    param_combs, combs = conf.param_combs, conf.combs
    combs_all = deepcopy(combs)



    if outdir is None:
        outdir = conf.outdir

    outpath = os.path.join(conf.outpath, outdir, "") #WRF output path
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    outpath_esc = outpath.replace("/", "\/") #need to escape slashes

    #temporary log output for job scheduler
    job_scheduler = None
    if use_job_scheduler:
        batch_log_dir = conf.run_path + "/logs/"
        os.makedirs(batch_log_dir, exist_ok=True)
        job_scheduler = conf.job_scheduler.lower()
        if job_scheduler not in ["slurm", "sge"]:
            raise ValueError("Job scheduler {} not implemented. Use SGE or SLURM".format(job_scheduler))
        if job_scheduler == "slurm":
            #assuming no oversubscription is allowed, pooling is necessary
            if conf.force_pool:
                pool_jobs = True
            mail_slurm = []
            for s,r in zip(["n", "b", "e", "a"], ["NONE", "BEGIN", "END", "FAIL"]):
                if s in mail:
                    mail_slurm.append(r)
            mail = ",".join(mail_slurm)
        if (conf.mail_address is None) or (conf.mail_address==""):
            raise ValueError("For jobs using {}, provide valid mail address in config file".format(job_scheduler))

    IDs = []
    rtr = []
    wrf_dir = []
    vmem = []
    nslots = []

    #%%
    if init:
        print("Initialize WRF simulations")
    else:
        if restart:
            print("Restart WRF simulations")
        else:
            print("Run WRF simulations")


    print("Configs:")
    print(pd.DataFrame(param_combs))
    print("-"*40)
    for i in range(len(combs)):
        args = deepcopy(pd.Series(combs[i]).dropna().to_dict())
        param_comb = param_combs[i]

        #create output ID for current configuration
        IDi, IDi_d = misc_tools.output_id_from_config(param_comb, conf.param_grid, conf.param_names, conf.runID)
        run_dir =  "{}/WRF_{}".format(conf.run_path, IDi)

        print("\n\nConfig:  " + IDi)
        print("\n".join(str(param_comb).split("\n")[:-1]))
        print("\n")

        r = args["dx"]

        #start and end times
        date_format = '%Y-%m-%d_%H:%M:%S'
        start_time_dt = datetime.datetime.strptime(args["start_time"],date_format)
        end_time_dt = datetime.datetime.strptime(args["end_time"],date_format)
        start_d, start_t = args["start_time"].split("_")
        start_d = start_d.split("-")
        start_t = start_t.split(":")
        end_d, end_t = args["end_time"].split("_")
        end_d = end_d.split("-")
        end_t = end_t.split(":")

        run_hours = (end_time_dt - start_time_dt).total_seconds()/3600

        for di,n in zip(start_d + start_t, ["year","month","day","hour","minute","second"] ):
            args["start_" + n] = di
        for di,n in zip(end_d + end_t, ["year","month","day","hour","minute","second"] ):
            args["end_" + n] = di
        #use end time not run_*
        args["run_hours"] = 0
        args["run_minutes"] = 0
        args["run_seconds"] = 0


        #hor. domain
        gp = conf.use_min_gridpoints
        fm = conf.force_domain_multiple
        if (not gp) or (gp == "y"):
            args["e_we"] = math.ceil(args["lx"]/r) + 1
        else:
            args["e_we"] = max(math.ceil(args["lx"]/r), args["min_gridpoints_x"] - 1) + 1
            if (fm == True) or (fm == "x"):
                lxr = (args["e_we"] -1)*r/args["lx"]
                if lxr != int(lxr):
                    raise Exception("Domain size must be multiple of lx")

        if (not gp) or (gp == "x"):
            args["e_sn"] = math.ceil(args["ly"]/r) + 1
        else:
            args["e_sn"] = max(math.ceil(args["ly"]/r), args["min_gridpoints_y"] - 1) + 1
            if (fm == True) or (fm == "y"):
                lyr = (args["e_sn"] -1)*r/args["ly"]
                if lyr != int(lyr):
                    raise Exception("Domain size must be multiple of ly")

        #slots
        if (conf.nslots_dict is not None) and (r in conf.nslots_dict) and (conf.nslots_dict[r] is not None):
            nx, ny = conf.nslots_dict[r]
        else:
            nx = misc_tools.find_nproc(args["e_we"]-1, min_n_per_proc=conf.min_n_per_proc, even_split=conf.even_split )
            ny = misc_tools.find_nproc(args["e_sn"]-1, min_n_per_proc=conf.min_n_per_proc, even_split=conf.even_split )
        if (nx == 1) and (ny == 1):
            nx = -1
            ny = -1
        if conf.max_nslotsx is not None:
            nx = min(conf.max_nslotsx, nx)
        if conf.max_nslotsy is not None:
            ny = min(conf.max_nslotsy, ny)
        nslotsi = nx*ny
        nslots.append(nslotsi)

        #add suffix for special folders
        slot_comm = ""
        if debug:
            wrf_dir_i = conf.wrf_dir_pre + "_debug"
        elif (nslotsi > 1) or (job_scheduler == "slurm"):
            wrf_dir_i = conf.wrf_dir_pre + "_mpi"
            if use_job_scheduler and (not pool_jobs):
                if job_scheduler == "sge":
                    slot_comm = "-pe openmpi-fillup {}".format(nslotsi)
                elif job_scheduler == "slurm":
                    slot_comm = "-N {}".format(nslotsi)
        else:
            wrf_dir_i = conf.wrf_dir_pre
        wrf_dir.append(wrf_dir_i)

        #timestep
        if "dt_f" not in args:
            args["dt_f"] = r/1000*6 #wrf rule of thumb

        if init:
            args, args_str, one_frame = misc_tools.prepare_init(args, conf, wrf_dir_i, namelist_check=not no_namelist_check)
            #job scheduler queue and vmem
            if use_job_scheduler and conf.request_vmem:
                vmem_init = max(conf.vmem_init_min, int(conf.vmem_init_per_grid_point*args["e_we"]*args["e_sn"]))
                if ("bigmem_limit" in dir(conf)) and (vmem_init > conf.bigmem_limit):
                    queue = conf.bigmem_queue
                else:
                    queue = conf.queue


        elif use_job_scheduler:
            queue = conf.queue
            args, skip = misc_tools.set_vmem_rt(args, run_dir, conf, run_hours, nslots=nslotsi, pool_jobs=pool_jobs, restart=restart, test_run=test_run, request_vmem=conf.request_vmem)
            if skip:
                continue
            if conf.request_vmem:
                vmemi = args["vmem"]
            else:
                vmemi = None
            vmem.append(vmemi)


        #not needed; just for completeness of dataframe:
        args["nx"] = nx
        args["ny"] = ny
        args["nslots"] = nslotsi
        args["run_dir"] = run_dir
        for arg, val in args.items():
            combs_all[i][arg] = val


        n_rep = args["n_rep"]
        for rep in range(n_rep): #repetion loop
            IDr = IDi + "_" + str(rep)
            IDs.append(IDr)
            run_dir_r = run_dir + "_" + str(rep)

            if init:
                if os.path.isdir(run_dir_r):
                    if exist == "s":
                        print("Run directory already exists.")
                        if os.path.isfile(run_dir_r + "/wrfinput_d01"):
                            print("Initialization was complete.\nSkipping...")
                            continue
                        else:
                            print("However, WRF initialization was not successfully carried out.\nRedoing initialization...")
                    elif exist == "o":
                        print("Overwriting...")
                    elif exist == "b":
                        print("Creating backup...")
                        bk_dir =  "{}/bak/".format(conf.run_path)
                        run_dir_bk =  "{}/WRF_{}_bak_".format(bk_dir, IDr)
                        os.makedirs(bk_dir, exist_ok=True)
                        bk_ind = 0
                        while os.path.isdir(run_dir_bk + str(bk_ind)):
                            bk_ind += 1
                        os.rename(run_dir_r, run_dir_bk + str(bk_ind))
                    else:
                        raise ValueError("Value '{}' for -e option not defined!".format(exist))

                hist_paths = r""
                for stream, (outfile, _) in args["output_streams"].items():
                    outname = r"{}{}_{}".format(outpath_esc, outfile, IDr)
                    if one_frame:
                        outname += "_<date>"

                    if stream == 0:
                        stream_arg = "history_outname"
                    else:
                        stream_arg = "auxhist{}_outname".format(stream)

                    hist_paths = r'''{} {} "{}"'''.format(hist_paths, stream_arg, outname)

                args_str_r = args_str + hist_paths

                iofile = ""
                if "iofields_filename" in args:
                    iofile_ = args["iofields_filename"].replace("'","").replace('"',"")
                    if iofile_ != "NONE_SPECIFIED":
                        iofile = iofile_
                comm_args =dict(JOB_NAME=IDr, wrfv=wrf_dir_i, ideal_case=conf.ideal_case, input_sounding=args["input_sounding"],
                                sleep=rep, nx=nx, ny=ny, run_path=conf.run_path, build_path=conf.build_path,
                                batch=int(use_job_scheduler), wrf_args="", cluster=int(conf.cluster), iofile=iofile, module_load=conf.module_load)
                for p, v in comm_args.items():
                    os.environ[p] = str(v)
                if use_job_scheduler:
                    os.environ["job_scheduler"] = job_scheduler
                    os.environ["wrf_args"] = args_str_r
                    #comm_args_str = ",".join(["{}='{}'".format(p,v) for p,v in comm_args.items()])
                    rt_init = misc_tools.format_timedelta(conf.rt_init*60)
                    qout, qerr = [batch_log_dir + IDr + s for s in [".out", ".err"]]
                    batch_args = [queue, qout, qerr, rt_init, conf.mail_address, mail, IDr]
                    if job_scheduler == "sge":
                        batch_args_str = "qsub -cwd -q {} -o {} -e {} -l h_rt={} -M {} -m {} -N {} -V ".format(*batch_args)
                        if "h_stack_init" in dir(conf) and conf.h_stack_init is not None:
                            batch_args_str += " -l h_stack={}M ".format(round(conf.h_stack_init))
                        if conf.request_vmem:
                            batch_args_str += " -l h_vmem={}M ".format(vmem_init)

                    elif job_scheduler == "slurm":
                        batch_args_str = "sbatch --qos={}  -p {} -o {} -e {} --time={} --mail-user={} --mail-type={} -J {} -N 1 -n 1 --export=ALL ".format(conf.qos, *batch_args)
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
                    print(initlog.split("\n")[-2].strip())

                    initerr = fopen(run_dir_r + "/init.err").read_text()
                    print(initerr)
                    if err != 0:
                        raise RuntimeError("Initialization failed!")



            else:
                skip = False
                if not os.path.isfile(run_dir_r + "/wrfinput_d01"):
                    print("Run not initialized yet! Skipping...")
                    skip = True
                stream_names = [stream[0] for stream in args["output_streams"].values()]
                send_rt_signal = conf.send_rt_signal
                if restart:
                    send_rt_signal = conf.send_rt_signal_restart
                    run_hours  = misc_tools.prepare_restart(run_dir_r, outpath, stream_names, args["end_time"])
                    if (run_hours is None) or (run_hours <= 0):
                        skip = True

                else:
                    outfiles = ["{}/{}_{}".format(outpath, outfile, IDr) for outfile in stream_names]
                    outfiles = [o for o in outfiles if os.path.isfile(o)]
                    if len(outfiles) > 0:
                        print("Output files already exist.")
                        if exist == "s":
                                print("Skipping...")
                                skip = True

                        elif exist == "o":
                            print("Overwriting...")
                        elif exist == "b":
                            print("Creating backup...")
                            bk_dir =  "{}/bak/".format(outpath)
                            os.makedirs(bk_dir, exist_ok=True)
                            for outfile in outfiles:
                                outfile_name = outfile.split("/")[-1]
                                outfile_bk = "{}/{}_bak_".format(bk_dir, outfile_name)
                                bk_ind = 0
                                while os.path.isfile(outfile_bk + str(bk_ind)):
                                    bk_ind += 1
                                os.rename(outfile, outfile_bk + str(bk_ind))
                        else:
                            raise ValueError("Value '{}' for -e option not defined!".format(exist))

                if skip:
                    IDs = []
                    rtr = []
                    vmem = []
                    nslots = []
                    wrf_dir = []
                    continue

                rtri = None
                if use_job_scheduler:
                    rtri = args["rt_per_timestep"] * run_hours/args["dt_f"] *3600 #runtime in seconds
                    rtr.append(rtri)

                last_id = False
                if (rep == n_rep-1) and (i == len(combs) - 1):
                    last_id = True

                if (not pool_jobs) or (sum(nslots) >= conf.pool_size) or last_id: #submit current pool of jobs
                    print("")
                    resched_i = False
                    #if pool is already too large: cut out last job, which is rescheduled afterwards
                    if pool_jobs and (sum(nslots) > conf.pool_size):
                        if len(nslots) > 1:
                            nslots = nslots[:-1]
                            wrf_dir = wrf_dir[:-1]
                            vmem = vmem[:-1]
                            rtr = rtr[:-1]
                            IDs = IDs[:-1]
                            resched_i = True
                        elif job_scheduler == "sge":
                            raise ValueError("Pool size ({}) smaller than number of slots of current job ({})!".format(conf.pool_size, nslots[0]))

                    iterate = True
                    while iterate:
                        iterate = False
                        print("Submit IDs: {}".format(IDs))
                        print("with total cores: {}".format(sum(nslots)))

                        if pool_jobs and use_job_scheduler:
                            job_name = "pool_" + "_".join(IDs)
                            if use_job_scheduler:
                                if job_scheduler == "sge":
                                    nperhost = conf.pool_size
                                    if conf.reduce_pool:
                                        #select smallest available nperhost that is greater or equal to the number of requested slots
                                        pes = os.popen("qconf -spl").read().split("\n")
                                        nperhost_available = np.array([int(pe[8:pe.index("per")]) for pe in pes if "perhost" in pe])
                                        nperhost = nperhost_available[nperhost_available >= sum(nslots)].min()
                                    slot_comm = "-pe openmpi-{0}perhost {0}".format(nperhost)
                                elif job_scheduler == "slurm":
                                    nodes = math.ceil(sum(nslots)/conf.pool_size)
                                    slot_comm = "--ntasks-per-node={} -N {}".format(conf.pool_size, nodes)

                        else:
                            job_name = IDr
                        wrf_dir = " ".join(wrf_dir)
                        jobs = " ".join(IDs)
                        nslots_str = " ".join([str(ns) for ns in nslots])
                        comm_args =dict(JOB_NAME=job_name, wrfv=wrf_dir, nslots=nslots_str, jobs=jobs, pool_jobs=int(pool_jobs), run_path=conf.run_path,
                                        batch=int(use_job_scheduler), cluster=int(conf.cluster), restart=int(restart), outpath=outpath, module_load=conf.module_load)
                        for p, v in comm_args.items():
                            os.environ[p] = str(v)
                        if use_job_scheduler:
                            os.environ["job_scheduler"] = job_scheduler

                            if conf.request_vmem:
                                vmemp = int(sum(vmem)/sum(nslots))
                                if ("bigmem_limit" in dir(conf)) and (vmemp > conf.bigmem_limit):
                                    queue = conf.bigmem_queue

                            rtr_max = max(rtr)
                            rtp = misc_tools.format_timedelta(rtr_max)
                            qout, qerr = [batch_log_dir + job_name + s for s in [".out", ".err"]]
                            if rtr_max < send_rt_signal:
                                raise ValueError("Requested runtime is smaller then the time when the runtime limit signal is sent!")

                            #comm_args_str = ",".join(["{}='{}'".format(p,v) for p,v in comm_args.items()])
                            batch_args = [conf.queue, qout, qerr, rtp, slot_comm, conf.mail_address, mail, job_name]
                            if job_scheduler == "sge":
                                batch_args_str = "qsub -cwd -q {} -o {} -e {} -l h_rt={}  {} -M {} -m {} -N {} -V ".format(*batch_args)
                                if "h_stack" in dir(conf) and conf.h_stack is not None:
                                    batch_args_str += " -l h_stack={}M ".format(round(conf.h_stack))
                                if conf.request_vmem:
                                    batch_args_str += " -l h_vmem={}M ".format(vmemp)
                                batch_args_str += " -l s_rt={} ".format(misc_tools.format_timedelta(rtr_max - send_rt_signal))
                            elif job_scheduler == "slurm":
                                batch_args_str = "sbatch -p {} -o {} -e {} --time={} {} --mail-user={} --mail-type={} -J {} --export=ALL ".format(*batch_args)
                                if  ("qos" in dir(conf)) and (conf.qos is not None):
                                    batch_args_str += " --qos={} ".format(conf.qos)
                                if conf.request_vmem:
                                    batch_args_str += " --mem-per-cpu={}M ".format(vmemp)
                                batch_args_str += " --signal=B:USR1@{} ".format(send_rt_signal)

                            comm = batch_args_str + " run_wrf.job"

                        else:
                            comm = "bash run_wrf.job"
                            if not wait:
                                comm += " &"

                        if verbose:
                            print(comm)
                        if not check_args:
                            err = os.system(comm)
                            if wait:
                                if restart:
                                    log_lines = 15
                                else:
                                    log_lines = 10

                                for ID in IDs:
                                    print(ID)
                                    run_dir_i = "{}/WRF_{}/".format(conf.run_path, ID)
                                    print(os.popen("tail -n {} {}".format(log_lines, run_dir_i + "run.log")).read())
                                    print(fopen(run_dir_i + "run.err").read_text())
                            if err != 0:
                                raise RuntimeError("WRF run failed!")

                        if resched_i:
                            IDs = [IDr]
                            rtr = [rtri]
                            vmem = [vmemi]
                            nslots = [nslotsi]
                            wrf_dir = [wrf_dir_i]

                        #run residual jobs that did not fit in the last job pool
                            if last_id:
                                iterate = True
                                last_id = False
                        else:
                            IDs = []
                            rtr = []
                            vmem = []
                            nslots = []
                            wrf_dir = []

    return pd.DataFrame(combs_all)

#%%
if __name__ == "__main__":

    # define argument parser


    sig = inspect.signature(submit_jobs).parameters

    #default function values
    defaults = {k : d.default for k, d in sig.items()}
    doc = submit_jobs.__doc__
    #get description from doc string
    desc = {}
    for k in defaults.keys():
        desc_k = doc[doc.index(" "+ k + " "):]
        desc[k] = desc_k.split("\n")[1].strip()

    intro = doc[:doc.index("Parameters\n")]

    #command line arguments (equivalent to function arguments above)
    #short form, long form, action
    parse_params = {"config_file":    ("-c", "--config", "store"),
                    "init":           ("-i", "--init", "store_true"),
                    "restart":        ("-r", "--restart", "store_true"),
                    "outdir":         ("-o", "--outdir", "store"),
                    "exist":          ("-e", "--exist", "store"),
                    "debug":          ("-d", "--debug", "store_true"),
                    "use_job_scheduler": ("-j", "--use_job_sched", "store_true"),
                    "check_args":     ("-t", "--test", "store_true"),
                    "pool_jobs":      ("-p", "--pool", "store_true"),
                    "mail":           ("-m", "--mail", "store"),
                    "wait":           ("-w", "--wait", "store_true"),
                    "no_namelist_check": ("-n", "--no_namelist_check", "store_false"),
                    "test_run":        ("-T", "--test_run", "store_true"),
                    "verbose":        ("-v", "--verbose", "store_true")
                    }

    assert sorted(parse_params.keys()) == sorted(desc.keys())
    add_args = {"exist" : {"choices" : ["s", "o", "b"]}}

    parser = argparse.ArgumentParser(description=intro)
    for p, default in defaults.items():
        if p in add_args:
            add_args_p = add_args[p]
        else:
            add_args_p = {}
        parser.add_argument(*parse_params[p][:-1], action=parse_params[p][-1], dest=p, default=default, help=desc[p], **add_args_p)

    parser.format_help()
    options = parser.parse_args()



    submit_jobs(**options.__dict__)

