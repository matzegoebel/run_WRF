#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:08:01 2019

Automatically initialize and run idealized experiments in WRF on a single computer or cluster.

@author: c7071088
"""
import numpy as np
import math
import os
import vertical_grid
import pandas as pd
import datetime
import argparse
import glob
import misc_tools
import importlib
import inspect

#%%
def submit_jobs(config_file="config", init=False, restart=False, outdir=None, exist="s", debug=False, use_qsub=False,
                check_args=False, pool_jobs=False, mail="ea", wait=False, verbose=False):
    """
    Submit idealized WRF experiments. Refer to README.md for more information.

    Parameters
    ----------
    config_file : TYPE, optional
        Name of config file in configs folder. The default is "config".
    init : TYPE, optional
        Initialize simulations.
    restart : TYPE, optional
        Restart simulations.
    outdir : TYPE, optional
        Subdirectory for WRF output. Default defined in script. Only effective during initialization.
    exist : TYPE, optional
        What to do if output already exists: Skip run ('s'), overwrite ('o') or backup files ('b'). Default is 's'.
    debug : TYPE, optional
        Run wrf in debugging mode. Just adds '_debug' to the build directory.
    use_qsub : TYPE, optional
        Use qsub to submit jobs
    check_args : TYPE, optional
        Only test python script (no jobs sumitted)
    pool_jobs : TYPE, optional
        Gather jobs before submitting with SGE. Needed if different jobs shall be run on the some node (potentially filling up the whole node)
    mail : TYPE, optional
        If using qsub, defines when mail is sent. Either 'n' for no mails, or a combination of 'b' (beginning of job), 'e' (end), 'a' (abort)', 's' (suspended). Default: 'ea'
    wait : TYPE, optional
        Wait until job is finished before submitting the next.
    verbose : TYPE, optional
        Verbose mode

    Returns
    -------
    combs : pandas DataFrame
        DataFrame with settings for all submitted configurations.

    """
    if (not init) and (outdir is not None):
        print("WARNING: option -o ignored when not in initialization mode!\n")
    if pool_jobs and (not use_qsub):
        raise ValueError("Pooling can only be used with --qsub option")
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

    if outdir is not None:
        conf.outdir = outdir

    conf.outpath = os.path.join(os.environ["wrf_res"], conf.outdir, "") #WRF output path
    if not os.path.isdir(conf.outpath):
        os.makedirs(conf.outpath)

    outpath_esc = conf.outpath.replace("/", "\/") #need to escape slashes

    IDs = []
    rtr = []
    wrf_dir = []
    vmem = []
    nslots = []

    #%%
    if init:
        print("Initialize WRF simulations")
    else:
        print("Run WRF simulations")

    print("Configs:")
    print(param_combs)
    print("-"*40)
    for i in range(len(combs)):
        args = combs.loc[i].dropna().to_dict()
        param_comb = param_combs.loc[i]

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
        args["run_hours"] = 0 #use end time not run_hours


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
        dt = args["dt"]
        #set a bunch of namelist parameters
        if init:
            args["dy"] = r
            args["dx"] = r

            if "isotropic_res" in args:
                if r <= args["isotropic_res"]:
                    args["mix_isotropic"] = 1
                else:
                    args["mix_isotropic"] = 0

            #timestep

            dt_int = math.floor(args["dt"])
            args["time_step"] = dt_int
            args["time_step_fract_num"] = round((args["dt"] - dt_int)*10)
            args["time_step_fract_den"] = 10
            if "radt" not in args:
                args["radt"] = max(args["radt_min"], 10*dt_int/60) #rule of thumb

            #vert. domain
            args["e_vert"] = args["nz"]
            eta, dz = vertical_grid.create_levels(nz=args["nz"], ztop=args["ztop"], method=args["dz_method"],
                                                              dz0=args["dz0"], plot=False, table=False)
            eta_levels = "'" + ",".join(eta.astype(str)) + "'"



            #split output in one timestep per file
            one_frame = False
            if r <= conf.split_output_res:
                args["frames_per_outfile"] = 1
                for out_ind in conf.output_streams.keys():
                    if out_ind != 0:
                        args["frames_per_auxhist{}".format(out_ind)] = 1
                one_frame = True

            #output streams
            for stream, (_, out_int) in conf.output_streams.items():
                out_int_m = math.floor(out_int)
                out_int_s = round((out_int - out_int_m)*60)
                if stream > 0:
                    stream_full = "auxhist{}".format(stream)
                else:
                    stream_full = "history"
                args["{}_interval_m".format(stream_full)] = out_int_m
                if out_int_s > 0:
                    args["{}_interval_s".format(stream_full)] = out_int_s

            #specified heat fluxes or land surface model
            if "spec_hfx" in args:
                args["ra_lw_physics"] = 0
                args["ra_sw_physics"] = 0
                args["tke_heat_flux"] = args["spec_hfx"]
                args["sf_surface_physics"] = 0
                if "isfflx" not in args:
                    args["isfflx"] = 2
            else:
                if "isfflx" not in args:
                    args["isfflx"] = 1
                args["tke_heat_flux"] = 0.
                args["tke_drag_coefficient"] = 0.

            #pbl scheme
            if r >= args["pbl_res"]:
                pbl_scheme = args["bl_pbl_physics"]
                if "km_opt" not in args:
                    args["km_opt"] = 4
            else:
                pbl_scheme = 0
                if "km_opt" not in args:
                    args["km_opt"] = 2

            args["bl_pbl_physics"] = pbl_scheme

            #choose surface layer scheme that is compatible with PBL scheme
            if "sf_sfclay_physics" not in args:
                if pbl_scheme in [1,2,3,4,5,7,10]:
                    sfclay_scheme = pbl_scheme
                elif pbl_scheme == 6:
                    sfclay_scheme = 5
                else:
                    sfclay_scheme = 1
                args["sf_sfclay_physics"] = sfclay_scheme

            # delete non-namelist parameters
            del_args = [*conf.del_args, *[p + "_idx" for p in conf.composite_params.keys()]]
            with open("{}/{}/test/{}/namelist.input".format(conf.build_path, wrf_dir_i, conf.ideal_case)) as namelist_file:
                namelist = namelist_file.read().replace(" ", "").replace("\t","")
            args_df = args.copy()
            for del_arg in del_args:
                if "\n"+del_arg+"=" in namelist:
                    raise RuntimeError("Parameter {} used in submit_jobs.py already defined in namelist.input! Rename this parameter!".format(del_arg))
                if del_arg in args_df:
                    del args_df[del_arg]

            args_df = pd.DataFrame(args_df, index=[0])

            #use integer datatype for integer parameters
            new_dtypes = {}
            for arg in args_df.keys():
                typ = args_df.dtypes[arg]
                if (typ == float) and (args_df[arg].dropna() == args_df[arg].dropna().astype(int)).all():
                    typ = int
                new_dtypes[arg] = typ
            args_dict = args_df.astype(new_dtypes).iloc[0].to_dict()

            args_str = " ".join(["{} {}".format(param, val) for param, val in args_dict.items()])

            if "iofields_filename" in args:
                args_str = args_str + """ iofields_filename "'{}'" """.format(args["iofields_filename"])
            args_str += " eta_levels " + eta_levels

            #vmem init and SGE queue
            vmem_init = max(conf.vmem_init_min, int(conf.vmem_init_per_grid_point*args["e_we"]*args["e_sn"]))
            if vmem_init > 25e3:
                init_queue = "bigmem.q"
            else:
                init_queue = "std.q"




        elif use_qsub:
            #get runtime per timestep
            identical_runs = None
            runtime_per_step = None
            if conf.rt is not None:
                runtime_per_step = conf.rt/3600*dt/run_hours #runtime per time step
            elif (conf.runtime_per_step_dict is not None) and (r in conf.runtime_per_step_dict):
                runtime_per_step = conf.runtime_per_step_dict[r]
                print("Use runtime dict")
            else:
                print("Get runtime from previous runs")
                run_dir_0 = run_dir + "_0" #use rep 0 as reference
                identical_runs = misc_tools.get_identical_runs(run_dir_0, conf.resource_search_paths)
                if len(identical_runs) > 0:
                    timing = misc_tools.get_runtime_all(runs=identical_runs, all_times=False)
                    if len(timing) > 0:
                        runtime_per_step, rt_sd = timing["timing"].mean(), timing["timing_sd"].mean()
                        print("Runtime per time step standard deviation: {0:.5f} s".format(rt_sd))

                if runtime_per_step is None:
                    print("No runtime specified and no previous runs found. Skipping...")
                    continue
            args["rt_per_timestep"] = runtime_per_step
            print("Runtime per time step: {0:.5f} s".format(runtime_per_step))

            #virtual memory
            vmemi = None
            if pool_jobs:
                vmemi = conf.vmem_pool
            elif conf.vmem is not None:
                vmemi = conf.vmem
            elif conf.vmem_per_grid_point is not None:
                print("Use vmem per grid point")
                vmemi = int(conf.vmem_per_grid_point*args["e_we"]*args["e_sn"])
                if conf.vmem_min is not None:
                    vmemi = max(vmemi, conf.vmem_min)
            else:
                print("Get vmem from previous runs")
                if identical_runs is None:
                    run_dir_0 = run_dir + "_0" #use rep 0 as reference
                    identical_runs = misc_tools.get_identical_runs(run_dir_0, conf.resource_search_paths)

                vmemi = misc_tools.get_vmem(identical_runs)
                if vmemi is None:
                    print("No vmem specified and no previous runs found. Skipping...")
                    continue
                else:
                    vmemi = max(vmemi)

            vmem.append(vmemi)
            args["vmem"] = vmemi
            print("Use vmem: {}".format(vmemi))

        #not needed; just for completeness of dataframe:
        args["nx"] = nx
        args["ny"] = ny
        args["nslots"] = nslotsi
        for arg, val in args.items():
            combs_all.loc[i, arg] = val


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
                for stream, (outfile, _) in conf.output_streams.items():
                    outname = r"{}{}_{}".format(outpath_esc, outfile, IDr)
                    if one_frame:
                        outname += "_<date>"

                    if stream == 0:
                        stream_arg = "history_outname"
                    else:
                        stream_arg = "auxhist{}_outname".format(stream)

                    hist_paths = r'''{} {} "{}"'''.format(hist_paths, stream_arg, outname)

                args_str = args_str + hist_paths
                comm_args =dict(wrfv=wrf_dir_i, ideal_case=conf.ideal_case, input_sounding=args["input_sounding"],
                                sleep=rep, nx=nx, ny=ny, wrf_args=args_str, run_path=conf.run_path, build_path=conf.build_path,
                                qsub=int(use_qsub), cluster=int(conf.cluster))
                if use_qsub:
                    comm_args_str = ",".join(["{}='{}'".format(p,v) for p,v in comm_args.items()])
                    comm = r"qsub -q {} -l h_vmem={}M -m {} -N {} -v {} init_wrf.job".format(init_queue, vmem_init, mail, IDr, comm_args_str)
                else:
                    for p, v in comm_args.items():
                        os.environ[p] = str(v)

                    os.environ["JOB_NAME"] = IDr
                    comm = "bash init_wrf.job '{}' ".format(args_str)

                if verbose:
                    print(comm)
                if not check_args:
                    err = os.system(comm)
                    ID_path = "{}/WRF_{}/".format(conf.run_path, IDr)
                    initlog = open(ID_path + "init.log").read()
                    print(initlog.split("\n")[-2].strip())
                    initerr = open(ID_path + "init.err").read()
                    print(initerr)
                    if err != 0:
                        raise RuntimeError("Initialization failed!")


            else:
                stream_names = [stream[0] for stream in conf.output_streams.values()]
                if restart:
                    wdir = "{}/WRF_{}/".format(conf.run_path,IDr)
                    run_hours = misc_tools.prepare_restart(wdir, IDr, conf.outpath, stream_names, end_time_dt)
                    if run_hours == 0:
                        continue
                else:
                    outfiles = ["{}/{}_{}".format(conf.outpath, outfile, IDr) for outfile in stream_names]
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
                            bk_dir =  "{}/bak/".format(conf.outpath)
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
                    rtri = runtime_per_step * run_hours/dt * conf.rt_buffer
                    rtr.append(rtri)

                last_id = False
                if (rep == n_rep-1) and (i == len(combs) - 1):
                    last_id = True

                if (sum(nslots) >= conf.pool_size) or (not pool_jobs) or conf.last_id: #submit current pool of jobs
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
                        wrf_dir = " ".join([str(wd) for wd in wrf_dir])
                        jobs = " ".join(IDs)
                        nslots = " ".join([str(ns) for ns in nslots])
                        comm_args =dict(wrfv=wrf_dir, nslots=nslots,jobs=jobs, pool_jobs=int(pool_jobs), run_path=conf.run_path,
                                        cluster=int(conf.cluster), wait=int(wait))
                        if use_qsub:
                            rtp = max(rtr)
                            rtp = "{:02d}:{:02d}:00".format(math.floor(rtp), math.ceil((rtp - math.floor(rtp))*60))
                            vmemp = int(sum(vmem)/len(vmem))
                            comm_args_str = ",".join(["{}='{}'".format(p,v) for p,v in comm_args.items()])
                            comm = r"qsub -q {} -N {} -l h_rt={} -l h_vmem={}M {} -m {} -v {} run_wrf.job".format(conf.queue, job_name, rtp, vmemp, slot_comm, mail, comm_args_str)
                        else:
                            for p, v in comm_args.items():
                                os.environ[p] = str(v)

                            comm = "bash run_wrf.job"

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
                        if verbose:
                            print(comm)
                        if not check_args:
                            err = os.system(comm)
                            if wait and err == 0:
                                rd = run_dir_r + "/"
                                if os.path.isfile(rd + "rsl.error.0000"):
                                    runlog = rd + "rsl.error.0000"
                                else:
                                    runlog = rd + "run.log"
                                tail_log = os.popen("tail -n 5 {}".format(runlog)).read()
                                print(tail_log)


    return combs_all

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

