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

#%%
def submit_jobs(config_file="config", init=False, restart=False, outdir=None, exist="s", debug=False, use_qsub=False,
           check_args=False, pool_jobs=False, mail="ea", wait=False, ignore_bad_namelist=False, verbose=False):
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
    use_qsub : bool, optional
        Use qsub to submit jobs
    check_args : bool, optional
        Only test python script (no jobs sumitted)
    pool_jobs : bool, optional
        Gather jobs before submitting with SGE. Needed if different jobs shall be run on the some node (potentially filling up the whole node)
    mail : str, optional
        If using qsub, defines when mail is sent. Either 'n' for no mails, or a combination of 'b' (beginning of job), 'e' (end), 'a' (abort)', 's' (suspended). Default: 'ea'
    wait : bool, optional
        Wait until job is finished before submitting the next.
    ignore_bad_namelist : bool, optional
        Ignore errors and warnings related to inconsistencies in namelist settings.
    verbose : bool, optional
        Verbose mode

    Returns
    -------
    combs : pandas DataFrame
        DataFrame with settings for all submitted configurations.

    """
    if (not init) and (outdir is not None):
        print("WARNING: option -o ignored when not in initialization mode!\n")
    if wait and use_qsub:
        raise ValueError("Waiting for SGE jobs is not yet implemented")
    if init and restart:
        raise ValueError("For restart runs no initialization is needed!")
    if len(glob.glob(os.getcwd() + "/submit_jobs.py")) == 0:
        raise RuntimeError("Script must be started from within its directory!")

    if config_file[-3:] == ".py":
        config_file = config_file[:-3]
    conf = importlib.import_module("configs.{}".format(config_file))
    param_combs, combs = conf.param_combs, conf.combs
    combs_all = combs.copy()

    if outdir is None:
        outdir = conf.outdir

    outpath = os.path.join(conf.outpath, outdir, "") #WRF output path
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    outpath_esc = outpath.replace("/", "\/") #need to escape slashes

    #temporary log output for SGE
    if use_qsub:
        sge_log_dir = conf.run_path + "/logs/"
        os.makedirs(sge_log_dir, exist_ok=True)

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
        args = pd.Series(combs[i]).dropna().to_dict()
        param_comb = param_combs[i]

        #create output ID for current configuration
        IDi, IDi_d = misc_tools.output_id_from_config(param_comb, conf.param_grid, conf.param_names, conf.runID)
        run_dir =  "{}/WRF_{}".format(conf.run_path, IDi)

        print("\n\nConfig:  " + IDi)
        print("\n".join(str(param_comb).split("\n")[:-1]))
        print("\n\n")

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
        elif nslotsi > 1:
            wrf_dir_i = conf.wrf_dir_pre + "_mpi"
            slot_comm = "-pe openmpi-fillup {}".format(nslotsi)
        else:
            wrf_dir_i = conf.wrf_dir_pre
        wrf_dir.append(wrf_dir_i)

        #timestep
        if "dt" not in args:
            args["dt"] = r/1000*6 #wrf rule of thumb

        if init:
            args, args_str, one_frame = misc_tools.prepare_init(args, conf, wrf_dir_i, ignore_bad_namelist=ignore_bad_namelist)

            #vmem init and SGE queue
            vmem_init = max(conf.vmem_init_min, int(conf.vmem_init_per_grid_point*args["e_we"]*args["e_sn"]))
            if vmem_init > 25e3:
                init_queue = "bigmem.q"
            else:
                init_queue = "std.q"

        elif use_qsub:
            args, skip = misc_tools.set_vmem_rt(args, run_dir, conf, run_hours, nslots=nslotsi, pool_jobs=pool_jobs)
            if skip:
                continue
            vmemi = args["vmem"]
            vmem.append(vmemi)
            print("Runtime per time step: {0:.5f} s".format(args["rt_per_timestep"]))
            print("Use vmem per slot: {}M".format(vmemi/nslotsi))

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
                comm_args =dict(wrfv=wrf_dir_i, ideal_case=conf.ideal_case, input_sounding=args["input_sounding"],
                                sleep=rep, nx=nx, ny=ny, wrf_args=args_str_r, run_path=conf.run_path, build_path=conf.build_path,
                                qsub=int(use_qsub), cluster=int(conf.cluster))
                if use_qsub:
                    comm_args_str = ",".join(["{}='{}'".format(p,v) for p,v in comm_args.items()])
                    rt_init = misc_tools.format_timedelta(conf.rt_init*60)
                    qout, qerr = [sge_log_dir + IDr + s for s in [".out", ".err"]]
                    sge_args = "-cwd -q {} -o {} -e {} -l h_rt={} -l h_vmem={}M -m {} -M {} -N {}".format(init_queue, qout, qerr, rt_init, vmem_init, mail, conf.mail_adress, IDr)
                    if "h_stack_init" in dir(conf) and conf.h_stack_init is not None:
                        sge_args += " -l h_stack={}M".format(round(conf.h_stack_init))

                    comm = r"qsub {} -v {} init_wrf.job".format(sge_args, comm_args_str)
                else:
                    for p, v in comm_args.items():
                        os.environ[p] = str(v)

                    os.environ["JOB_NAME"] = IDr
                    comm = "bash init_wrf.job '{}' ".format(args_str_r)

                if verbose:
                    print(comm)
                if not check_args:
                    err = os.system(comm)
                    if err == 0:
                        initlog = open(run_dir_r + "/init.log").read()
                        print(initlog.split("\n")[-2].strip())
                    else:
                        initerr = open(run_dir_r + "/init.err").read()
                        print(initerr)
                        raise RuntimeError("Initialization failed!")


            else:
                if not os.path.isfile(run_dir_r + "/wrfinput_d01"):
                    print("Run not initialized yet! Skipping...")
                    continue
                stream_names = [stream[0] for stream in args["output_streams"].values()]
                if restart:
                    args["rt_per_timestep"]  = misc_tools.prepare_restart(run_dir_r, outpath, stream_names, args["end_time"])
                    if args["rt_per_timestep"]  <= 0:
                        print("Run already complete")
                        continue
                    elif run_hours is None:
                        continue
                else:
                    outfiles = ["{}/{}_{}".format(outpath, outfile, IDr) for outfile in stream_names]
                    outfiles = [o for o in outfiles if os.path.isfile(o)]
                    if len(outfiles) > 0:
                        print("Output files already exist.")
                        if exist == "s":
                                print("Skipping...")
                                continue
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



                rtri = None
                if use_qsub:
                    rtri = args["rt_per_timestep"] * run_hours/args["dt"] * conf.rt_buffer
                    rtr.append(rtri)

                last_id = False
                if (rep == n_rep-1) and (i == len(combs) - 1):
                    last_id = True

                if (sum(nslots) >= conf.pool_size) or (not pool_jobs) or last_id: #submit current pool of jobs
                    print("")
                    resched_i = False
                    #if pool is already too large: cut out last job, which is rescheduled afterwards
                    if pool_jobs and (sum(nslots) > conf.pool_size):
                        if len(nslots) == 1:
                            raise ValueError("Pool size ({}) smaller than number of slots of current job ({})!".format(conf.pool_size, nslots[0]))
                        nslots = nslots[:-1]
                        wrf_dir = wrf_dir[:-1]
                        vmem = vmem[:-1]
                        rtr = rtr[:-1]
                        IDs = IDs[:-1]
                        resched_i = True

                    iterate = True
                    while iterate:
                        iterate = False
                        print("Submit IDs: {}".format(IDs))
                        print("with total cores: {}".format(sum(nslots)))

                        if pool_jobs and use_qsub:
                            job_name = "pool_" + "_".join(IDs)
                            if use_qsub:
                                nperhost =  conf.pool_size
                                if conf.reduce_pool:
                                    nperhost_sge = np.array([1, *np.arange(2,29,2)])
                                    nperhost = nperhost_sge[(nperhost_sge >= sum(nslots)).argmax()] #select available nperhost that is greater and closest to the number of slots
                                slot_comm = "-pe openmpi-{0}perhost {0}".format(nperhost)
                        else:
                            job_name = IDr
                        wrf_dir = " ".join(wrf_dir)
                        jobs = " ".join(IDs)
                        nslots_str = " ".join([str(ns) for ns in nslots])
                        comm_args =dict(wrfv=wrf_dir, nslots=nslots_str, jobs=jobs, pool_jobs=int(pool_jobs), run_path=conf.run_path,
                                        cluster=int(conf.cluster), wait=int(wait), restart=int(restart), outpath=outpath)
                        if use_qsub:
                            vmemp = int(sum(vmem)/sum(nslots))
                            rtp = misc_tools.format_timedelta(max(rtr)*3600)
                            qout, qerr = [sge_log_dir + job_name + s for s in [".out", ".err"]]
                            sge_args = "-cwd -q {} -o {} -e {} -N {} -l h_rt={} -l h_vmem={}M {} -m {} -M {}".format(conf.queue, qout, qerr, job_name, rtp, vmemp, slot_comm, mail, conf.mail_adress)
                            if "h_stack" in dir(conf) and conf.h_stack is not None:
                                sge_args += " -l h_stack={}M".format(round(conf.h_stack))

                            comm_args_str = ",".join(["{}='{}'".format(p,v) for p,v in comm_args.items()])
                            comm = r"qsub {} -v {} run_wrf.job".format(sge_args, comm_args_str)

                        else:
                            for p, v in comm_args.items():
                                os.environ[p] = str(v)

                            comm = "bash run_wrf.job"

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
                                    print(open(run_dir_i + "run.err").read())
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
                    "use_qsub":       ("-q", "--qsub", "store_true"),
                    "check_args":     ("-t", "--test", "store_true"),
                    "pool_jobs":      ("-p", "--pool", "store_true"),
                    "mail":           ("-m", "--mail", "store"),
                    "wait":           ("-w", "--wait", "store_true"),
                    "ignore_bad_namelist":  ("-n", "--ignore_bad_namelist", "store_true"),
                    "verbose":        ("-v", "--verbose", "store_true")
                    }

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

