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


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config",
                    action="store", dest="config_file", default = "config",
                    help="Name of config file in configs folder.")
parser.add_argument("-i", "--init",
                    action="store_true", dest="init", default = False,
                    help="Initialize simulations")
parser.add_argument("-r", "--restart",
                    action="store_true", dest="restart", default = False,
                    help="Restart simulations")
parser.add_argument("-o", "--outdir",
                    action="store", dest="outdir", default = None,
                    help="Subdirectory for WRF output. Default defined in script. Only effective during initialization.")
parser.add_argument("-d", "--debug",
                    action="store_true", dest="debug", default = False,
                    help="Run wrf in debugging mode. Just adds '_debug' to the build directory.")
parser.add_argument("-q", "--qsub",
                    action="store_true", dest="use_qsub", default = False,
                    help="Use qsub to submit jobs")
parser.add_argument("-t", "--test",
                    action="store_true", dest="check_args", default = False,
                    help="Only test python script (no jobs sumitted)")
parser.add_argument("-T", "--check_runtime",
                    action="store_true", dest="check_rt", default = False,
                    help="Start short test runs to determine runtime of simulations, for which no identical simulations exist yet (only with qsub option)")
parser.add_argument("-p", "--pool",
                    action="store_true", dest="pool_jobs", default = False,
                    help="Gather jobs before submitting with SGE. Needed if different jobs shall be run on the some node (potentially filling up the whole node)")
parser.add_argument("-m", "--mail",
                    action="store", dest="mail", type=str, default = "ea",
                    help="If using qsub, defines when mail is sent. Either 'n' for no mails, or a combination of 'b' (beginning of job), 'e' (end), 'a' (abort)', 's' (suspended). Default: 'ea'")
parser.add_argument("-v", "--verbose",
                    action="store_true", dest="verbose",default=False,
                    help="Verbose mode")


options = parser.parse_args()
if (not options.init) and (options.outdir is not None):
    print("WARNING: option -o ignored when not in initialization mode!\n")

if options.pool_jobs and (not options.use_qsub):
    raise ValueError("Pooling can only be used with --qsub option")
if options.check_rt and (not options.use_qsub):
    raise ValueError("Runtime checking can only be used with --qsub option")
if options.check_rt and options.init:
    raise ValueError("Runtime checking cannot be used in initialization mode")
if options.check_rt and options.pool_jobs:
    raise ValueError("Runtime checking cannot be used with pooling")
if options.check_rt and options.restart:
    raise ValueError("Runtime checking cannot be used with restart runs")
if len(glob.glob(os.getcwd() + "/submit_jobs.py")) == 0:
    raise RuntimeError("Script must be started from within its directory!")

#%%
conf = importlib.import_module("configs.{}".format(options.config_file))
param_combs, param_grid_flat, composite_params = misc_tools.grid_combinations(conf.param_grid)

if options.outdir is not None:
    conf.outdir = options.outdir
    
conf.outpath = os.path.join(os.environ["wrf_res"], conf.outdir, "") #WRF output path
if not os.path.isdir(conf.outpath):
    os.makedirs(conf.outpath)
    
outpath_esc = conf.outpath.replace("/", "\/") #need to escape slashes
if options.restart:
    init = False

start_time_dt = datetime.datetime.fromisoformat(conf.start_time)
end_time_dt = datetime.datetime.fromisoformat(conf.end_time)
start_d, start_t = conf.start_time.split("_")
start_d = start_d.split("-")
start_t = start_t.split(":")
end_d, end_t = conf.end_time.split("_")
end_d = end_d.split("-")
end_t = end_t.split(":")

run_hours = int((end_time_dt - start_time_dt).total_seconds()/3600)

IDs = []
rtr = []
wrf_dir = []
vmem = []
nslots = []
dx_p = []
pi = 0

for di,n in zip(start_d + start_t, ["year","month","day","hour","minute","second"] ):
    conf.params["start_" + n] = di

if "dx" in param_grid_flat:
    dx = param_grid_flat["dx"]
else:
    dx = [conf.params["dx"]]

#combine param grid and additional settings
combs = param_combs.copy()
for param, val in conf.params.items():
    if param not in combs:
        combs[param] = val


#%%
if options.init:
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
    IDi = param_comb.copy()
    for p, v in conf.param_grid.items():
        if type(v) == dict:
            for pc in v.keys():
                del IDi[pc]
        if p in conf.param_names:
            namep = conf.param_names[p]
            if type(v) == dict:
                IDi[p] = namep[IDi[p + "_idx"]]
                del IDi[p + "_idx"]
            else:
                if type(namep) == dict:
                    IDi[p] = namep[IDi[p]]
                else:
                    IDi[p] = namep[conf.param_grid[p].index(IDi[p])]


    IDi = "_".join(IDi.values.astype(str))
    IDi =  conf.runID + "_" + IDi

    print("\n\nConfig:")
    print("\n".join(str(param_comb).split("\n")[:-1]))

    r = args["dx"]
    #dx_p.append(r)
    args["dy"] = r
    args["dx"] = r
    if args["topo"] == "flat":
        args["topo"] = 0
    elif args["topo"] == "cos":
        args["topo"] = 2
    else:
        raise Exception("type of topo not known!")

    if r <= args["isotropic_res"]:
        args["mix_isotropic"] = 1
    else:
        args["mix_isotropic"] = 0

    args["run_hours"] = run_hours
    if "dt" not in args:
        args["dt"] = r/1000*6 #wrf rule of thumb
    dt = args["dt"]

    dt_int = math.floor(args["dt"])
    args["time_step"] = dt_int
    args["time_step_fract_num"] = round((args["dt"] - dt_int)*10)
    args["time_step_fract_den"] = 10
    radt =  max(1, 10*dt_int/60)
    args["radt"] = radt

    if conf.use_gridpoints:
        args["e_we"] = max(math.ceil(args["lx"]/r), args["gridpoints"]) + 1
        lxr = (args["e_we"] -1)*r/args["lx"]
        if conf.force_domain_multiple:
            if lxr != int(lxr):
                raise Exception("Domain size must be multiple of lx")
        args["e_sn"] = max(math.ceil(args["ly"]/r), args["gridpoints"]) + 1
    else:
        args["e_we"] = math.ceil(args["lx"]/r) + 1
        args["e_sn"] = math.ceil(args["ly"]/r) + 1

    vmem_init = max(conf.vmem_init_min, int(conf.vmem_init_per_grid_point*args["e_we"]*args["e_sn"]))

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

    args["e_vert"] = args["nz"]
    eta, dz = vertical_grid.create_levels(nz=args["nz"], ztop=args["ztop"], method=args["dz_method"],
                                                      dz0=args["dz0"], plot=False, table=False)

    eta_levels = "'" + ",".join(eta.astype(str)) + "'"
    one_frame = False
    if r <= conf.split_output_res: #split output in one timestep per file
        args["frames_per_outfile"] = 1
        for out_ind in conf.output_streams.keys():
            if out_ind != 0:
                args["frames_per_auxhist{}".format(out_ind)] = 1
        one_frame = True

    for stream, (_, out_int) in conf.output_streams.items():
        if stream > 0:
            args["auxhist{}_interval".format(stream)] = out_int
        else:
            args["history_interval_m"] = out_int
            args["history_interval"] = out_int

    if options.init:
        if "spec_hfx" in args:
            args["ra_lw_physics"] = 0
            args["ra_sw_physics"] = 0
            args["tke_heat_flux"] = args["spec_hfx"]
            args["sf_surface_physics"] = 0
            args["isfflx"] = 2
        elif "spec_sw" in args:
            args["swin"] = args["spec_sw"]
            args["const_sw"] = True
        else:
            args["const_sw"] = False
            args["isfflx"] = 1
            args["tke_heat_flux"] = 0.
            args["tke_drag_coefficient"] = 0.

        if r >= args["pbl_res"]:
            if ("iofields_filename" in args) and (args["iofields_filename"]==0):
                args["iofields_filename"] = '"MESO_IO.txt"'
            pbl_scheme = args["bl_pbl_physics"]
            if "km_opt" not in args:
                args["km_opt"] = 4
        else:
            if ("iofields_filename" in args) and (args["iofields_filename"]==0):
                args["iofields_filename"] = '"LES_IO.txt"'
            pbl_scheme = 0

            if "km_opt" not in args:
                args["km_opt"] = 2

        args["bl_pbl_physics"] = pbl_scheme
        if pbl_scheme in [5, 6]:
            args["bl_mynn_tkebudget"] = 1
        else:
            args["bl_mynn_tkebudget"] = 0
            args["bl_mynn_edmf"] = 0
            args["bl_mynn_edmf_tke"] = 0
            args["scalar_pblmix"] = 0

        #choose surface layer scheme that is compatible with PBL scheme
        if "sf_sfclay_physics" not in args:
            if pbl_scheme in [1,2,3,4,5,7,10]:
                sfclay_scheme = pbl_scheme
            elif pbl_scheme == 6:
                sfclay_scheme = 5
            else:
                sfclay_scheme = 1


            args["sf_sfclay_physics"] = sfclay_scheme

        if "init_pert" not in args:
            if conf.no_pert:
                args["init_pert"] = False
            elif conf.all_pert or (r <= args["pert_res"]):
                args["init_pert"] = True
            else:
                args["init_pert"] = False

        # delete non-namelist parameters
        del_args =   ["dx", "nz", "dz0","dz_method", "gridpoints", "lx", "ly", "spec_hfx", "spec_sw",
                    "pert_res", "input_sounding", "repi", "n_rep", "isotropic_res", "pbl_res", "dt",
                    *[p + "_idx" for p in composite_params.keys()]]
        args_df = args.copy()
        for del_arg in del_args:
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
        args_str = args_df.astype(new_dtypes).iloc[0].to_dict()

        #add a few more parameters
        args_str["init_pert"] = misc_tools.bool_to_fort(args_str["init_pert"])
        args_str["const_sw"] = misc_tools.bool_to_fort(args_str["const_sw"])
        args_str = str(args_str).replace("{","").replace("}","").replace("'","").replace(":","").replace(",","")
        if "iofields_filename" in args:
            args_str = args_str +  " iofields_filename " + args["iofields_filename"]
        args_str += " eta_levels " + eta_levels

        if vmem_init > 25e3:
            init_queue = "bigmem.q"
        else:
            init_queue = "std.q"

        args["nslots"] = nslotsi

    elif options.use_qsub:
        runtime_per_step = None
        if conf.rt is not None:
            runtime_per_step = conf.rt/3600*dt/run_hours #runtime per time step
        elif (conf.runtime_per_step_dict is not None) and (r in conf.runtime_per_step_dict):
            runtime_per_step = conf.runtime_per_step_dict[r]
        
        if options.check_rt or (runtime_per_step is None):
            print("Search for runtime values in previous runs.")
            timing = misc_tools.get_runtime_all(IDi,conf.rt_search_paths, all_times=False, levels=len(param_combs.keys())+2)
            valid_ids = []
            for i_p in range(len(timing)):
                p = timing.loc[i_p, "path"]
                namelist_diff = os.popen("diff -B -w  --suppress-common-lines -y {}/namelist.input\
                                         {}/WRF_{}_{}/namelist.input".format(p, conf.run_path, IDi, args["repi"])).read()
                namelist_diff = namelist_diff.replace(" ","").replace("\t","").split("\n")
                important_diffs = []
                for diff in namelist_diff:
                    if diff == "":
                        continue
                    old, new = diff.split("|")
                    if  (old[0] == "!") and (new[0] == "!"):
                        continue
                    for ignore_param in ["start_year", "start_month","start_day", "start_hour",
                                          "start_minute","run_hours", "_outname"]:
                        if ignore_param + "=" in diff:
                            continue
                    important_diffs.append(diff)
                if len(important_diffs) > 0:
                    print("{} has different namelist parameters".format(p))
                else:
                    valid_ids.append(i_p)
                    print("{} has same namelist parameters".format(p))
            if len(valid_ids) > 0:
                runtime_per_step = timing.iloc[valid_ids]["timing"].mean()
                if options.check_rt:
                    print("Previous runs found. No test run needed.")
                else:
                    print("Use runtime per time step: {0:.5f} s ".format(runtime_per_step))

            elif options.check_rt:
                print("No valid previous runs found. Do test run.")
                args["n_rep"] = 1
            else:
                print("No runtime specified and no previous runs found. Skipping...")
                continue
        args["rt_per_timestep"] = runtime_per_step

    args["nx"] = nx
    args["ny"] = ny
    for arg, val in args.items():
        combs.loc[i, arg] = val #not needed; just for completeness of dataframe

    if options.debug:
        wrf_dir_i = conf.wrf_dir_pre + "_debug"
        slot_comm = ""
    elif nslotsi > 1:
        wrf_dir_i = conf.wrf_dir_pre + "_mpi"
        slot_comm = "-pe openmpi-fillup {}".format(nslotsi)
    else:
        wrf_dir_i = conf.wrf_dir_pre
        slot_comm = ""

    wrf_dir.append(wrf_dir_i)
    if options.pool_jobs:
        vmemi = conf.vmem_pool
    else:
        vmemi = max(conf.vmem_min, int(conf.vmem_per_grid_point*args["e_we"]*args["e_sn"]))
    vmem.append(vmemi)


    repi = args["repi"]
    n_rep = args["n_rep"]
    for rep in range(repi, n_rep+repi): #repetion loop
        IDr = IDi + "_" + str(rep)
        IDs.append(IDr)

        if options.init:
            print("")

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
                            sleep=rep, nx=nx, ny=ny, wrf_args=args_str, run_path=conf.run_path, build_path=conf.build_path,qsub=int(options.use_qsub))
            if options.use_qsub:
                comm_args_str = ",".join(["{}='{}'".format(p,v) for p,v in comm_args.items()])
                comm = r"qsub -q {} -l h_vmem={}M -m {} -N {} -v {} init_wrf.job".format(init_queue, vmem_init, options.mail, IDr, comm_args_str)
            else:
                d = {}
                for p, v in comm_args.items():
                    os.environ[p] = str(v)

                os.environ["JOB_NAME"] = IDr
                comm = "bash init_wrf.job '{}' ".format(args_str)

            if options.verbose:
                print(comm)
            if not options.check_args:
                os.system(comm)
                ID_path = "{}/WRF_{}".format(conf.run_path, IDr)
                os.system("tail -n 1 {}/init.log".format(ID_path))
                os.system("cat {}/init.err".format(ID_path))

        else:
            if options.restart:
                wdir = "{}/WRF_{}/".format(conf.run_path,IDr)
                rstfiles = os.popen("ls -t {}/wrfrst*".format(wdir)).read()
                if rstfiles == "":
                    print("WARNING: no restart files found")

                restart_time = rstfiles.split("\n")[0].split("/")[-1].split("_")[-2:]
                start_time_rst = datetime.datetime.fromisoformat("_".join(restart_time))
                end_time_rst = datetime.datetime.fromisoformat(conf.end_time)
                rst_date, rst_time = restart_time
                rst_date = rst_date.split("-")
                rst_time = rst_time.split(":")
                run_hours = int((end_time_rst - start_time_rst).total_seconds()/3600)
                if run_hours == 0:
                    continue
                rst_opt = "restart .true. start_year {} start_month {} start_day {}\
                start_hour {} start_minute {} start_second {} run_hours {}".format(*rst_date, *rst_time, conf.run_h)
                os.makedirs("{}/bk/".format(conf.outpath), exist_ok=True) #move previous output in backup directory
                bk_files = glob.glob("{}/bk/*{}".format(conf.outpath, IDr))
                if len(bk_files) != 0:
                    raise FileExistsError("Backup files for restart already present. Double check!")
                os.system("mv {}/*{} {}/bk/".format(conf.outpath, IDr, conf.outpath))
                os.system("bash search_replace.sh {0}/namelist.input {0}/namelist.input {1}".format(wdir, rst_opt))
            
            rtri = None
            if options.use_qsub:
                if options.check_rt:
                    rtri = conf.rt_check/3600
                else:   
                    rtri = runtime_per_step * run_hours/dt * conf.rt_buffer
                    rtr.append(rtri)

            #split_res = False
            last_id = False
 #           dx_p_set = [str(int(rs)) for rs in set(dx_p)]
#            if (np.in1d(dx_ind, dx_p).any()) and (len(dx_p_set) > 1):
#                split_res = True
            if (rep == n_rep+repi-1) and (i == len(combs) - 1):
                last_id = True

            if (sum(nslots) >= conf.pool_size) or (not options.pool_jobs) or conf.last_id: #submit current pool of jobs
                print("")
                resched_i = False
                #if pool is already too large: cut out last job, which is rescheduled afterwards
                if (sum(nslots) > conf.pool_size):
                    if len(nslots) == 1:
                        raise ValueError("Pool size ({}) smaller than number of slots of current job ({})!".format(conf.pool_size, nslots[0]))
                    nslots = nslots[:-1]
                    wrf_dir = wrf_dir[:-1]
                    vmem = vmem[:-1]
                    rtr = rtr[:-1]
                    IDs = IDs[:-1]
                  #  dx_p = dx_p[:-1]
                    resched_i = True
                #    dx_p_set = [str(int(rs)) for rs in set(dx_p)]

                iterate = True
                while iterate:
                    iterate = False
                    print("Submit IDs: {}".format(IDs))
                    print("with total cores: {}".format(sum(nslots)))

                    if options.pool_jobs:
                        nperhost =  conf.pool_size
                        if conf.reduce_pool:
                            nperhost_sge = np.array([1, *np.arange(2,29,2)])
                            nperhost = nperhost_sge[(nperhost_sge >= sum(nslots)).argmax()] #select available nperhost that is greater and closest to the number of slots
                        slot_comm = "-pe openmpi-{0}perhost {0}".format(nperhost)

                    wrf_dir = " ".join([str(wd) for wd in wrf_dir])
                    if options.pool_jobs:
                        job_name = "pool_" + "_".join(IDs)
                    else:
                        job_name = IDr
                    jobs = " ".join(IDs)

                    comm_args =dict(wrfv=wrf_dir, nslots=nslots,jobs=jobs, pool_jobs=int(options.pool_jobs), run_path=conf.run_path, cluster=int(conf.cluster))
                    if options.use_qsub:
                        rtp = max(rtr)
                        rtp = "{:02d}:{:02d}:00".format(math.floor(rtp), math.ceil((rtp - math.floor(rtp))*60))
                        nslots = " ".join([str(ns) for ns in nslots])
                        vmemp = int(sum(vmem)/len(vmem))
                        comm_args_str = ",".join(["{}='{}'".format(p,v) for p,v in comm_args.items()])
                        comm = r"qsub -q {} -N {} -l h_rt={} -l h_vmem={}M {} -m {} -v {} run_wrf.job".format(conf.queue, job_name, rtp, vmemp, slot_comm, options.mail, comm_args_str)
                    else:
                        for p, v in comm_args.items():
                            os.environ[p] = str(v)

                        comm = "bash run_wrf.job"

                    pi = 0
                    if resched_i:
                        IDs = [IDr]
                        rtr = [rtri]
                        vmem = [vmemi]
                        nslots = [nslotsi]
                        wrf_dir = [wrf_dir_i]
                       # dx_p = [r]
                        if last_id:
                            iterate = True
                            last_id = False
                    else:
                        IDs = []
                        rtr = []
                        vmem = []
                        nslots = []
                        wrf_dir = []
                        dx_p = []
                    if options.verbose:
                        print(comm)
                    if not options.check_args:
                        os.system(comm)

            else:
                pi += 1
