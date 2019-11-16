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
import misc
from collections import OrderedDict as odict


parser = argparse.ArgumentParser()
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
parser.add_argument("-p", "--pool",
                    action="store_true", dest="pool_jobs", default = False,
                    help="Gather jobs before submitting with SGE. Needed if different jobs shall be run on the some node (potentially filling up the whole node)")
parser.add_argument("-m", "--mail",
                    action="store", dest="mail", type=str, default = "ea",
                    help="If using qsub, defines when mail is sent. Either 'n' for no mails, or a combination of 'b' (beginning of job), 'e' (end), 'a' (abort)', 's' (suspended). Default: 'ea'")


options = parser.parse_args()
if (not options.init) and (options.outdir is not None):
    print("WARNING: option -o ignored when not in initialization mode!\n")

if options.pool_jobs and (not options.use_qsub):
    raise ValueError("Pooling can only be used with --qsub option")
#%%
'''Simulations settings'''

wrf_dir_pre = "WRF" #prefix for WRF build directory (_debug and _mpi will be added later)
ideal_case = "em_les" #idealized WRF case
runID = "test" #name for this simulation series

outdir = "test/qbudget" #subdirectory for WRF output if not set in command line
if options.outdir is not None:
    outdir = options.outdir

outpath = os.path.join(os.environ["wrf_res"], outdir, "") #WRF output path
run_path = os.environ["wrf_runs"] #path where run directories of simulations will be created
build_path = os.environ["wrf_builds"]

#Define parameter grid for simulations (any namelist parameters and some additional ones can be used)

# param_grid = dict(
#              dx=[4000, 2000, 1000, 500, 250, 125, 62.5],# hor. grid spacings in meter
#             input_sounding=["schlemmer_stable","schlemmer_unstable"],
#             topo=["cos", "flat"])

#param_grid = odict(
#            c1={"dx" : [125,4000,4000,4000], "bl_pbl_physics": [0,1,5,7], "dz0" : [10,50,50,50], "nz" : [350,60,60,60], "composite_name": ["LES", "MYJ", "MYNN", "ACM2"]})
param_grid = odict(
            c1={"dx" : [4000], "bl_pbl_physics": [7], "dz0" : [50], "nz" : [60], "composite_name": ["ACM2"]})

param_combs, param_grid_flat, composite_params = misc.grid_combinations(param_grid)
a = {}

#Set additional namelist parameters (only applies if they are not present in param_grid)
#any namelist parameters and some additional ones can be used

start_time = "2018-06-20_00:00:00" #"2018-06-20_20:00:00"
end_time = "2018-06-21_00:00:00" #"2018-06-23_00:00:00"

a["n_rep"] = 1 #number of repetitions for each configuration
a["repi"] = 0#start id for reps

#horizontal grid
a["dx"] = 500 #horizontal grid spacing (m)
a["lx"] = 50 #16000, horizontal extent in east west (m)
a["ly"] = 50 #4000, minimum horizontal extent in north south (m)
use_gridpoints = True #use minimum number of grid points set below
a["gridpoints"] = 2 #16, minimum number of grid points in each direction -1
force_domain_multiple = True #if use_gridpoints: force domain with x and y extents that are multiples of lx and ly, respectively

#vertical grid
a["ztop"] = 15000 #15000, top of domain (m)
a["zdamp"] = int(a["ztop"]/3) #depth of damping layer (m)
a["nz"] = 122 #176, number of vertical levels
a["dz0"] = 20 #10, height of first model level (m)
a["dz_method"] = 0 #method for creating vertical grid as defined in vertical_grid.py
a["dt"] = None #1 #time step (s), if None calculated as dt = 6 s/m *dx/1000

a["input_sounding"] = "schlemmer_stable" #name of input sounding to use (should be named input_sounding_name)

a["isotropic_res"] = 100 #resolution below which mixing is isotropic
a["pbl_res"] = 500 #resolution above which to use PBL scheme (m)
a["spec_hfx"] = None #None specified surface heat flux instead of radiation

#standard namelist parameters
a["mp_physics"] = 2
a["bl_pbl_physics"] = 6
a["bl_mynn_edmf"] = 1
a["bl_mynn_edmf_tke"] = 1
a["scalar_pblmix"] = 1
a["topo_shading"] = 1
a["slope_rad"] = 1

#custom namelist parameters (not available in official WRF)
a["topo"] = "flat"#, "cos"] #topography type
a["spec_sw"] = None  # specified constant shortwave radiation
a["pert_res"] = 4000 #resolution below which initial perturbations are used
no_pert = False
all_pert = False

#indices for output streams and their respective name and output interval (min); 0 is the standard output stream
output_streams = {0: ["wrfout", 30], 7: ["fastout", 10], 8 : ["meanout", 30]}


#%%
'''Settings for resource requirements of SGE jobs'''
cluster_name = "leo" #this name should appear in the variable $HOSTNAME to detect if cluster settings should be used
queue = "std.q" #queue for SGE

split_output_res = 0 #resolution below which to split output in one timestep per file

#virtual memory: numbers need adjustment
#TODO: approach is not optimized yet!
vmem_init_per_grid_point = 0.3 #virtual memory (MB) per horizontal grid point to request for WRF initialization (ideal.exe)
vmem_init_min = 2000 #minimum virtual memory (MB) for WRF initialization

vmem_per_grid_point = 0.3 #virtual memory (MB) per horizontal grid point to request for running WRF (wrf.exe); will be divided by number of slots
vmem_min = 600 #minimum virtual memory (MB) for running WRF
vmem_pool = 2000 #virtual memory to request per slot if pooling is used

vmem_buffer = 1.2 #buffer factor for virtual memory

# runtime 
#TODO: make gridpoint dependent or so; make second res
rt = None #None or job runtime in seconds
rt_buffer = 1.5 #buffer factor to multiply rt with
runtime_per_step = { 62.5 : 3., 100: 3., 125. : 3. , 250. : 1., 500: 0.5, 1000: 0.3, 2000.: 0.3, 4000.: 0.3}# if rt is None: runtime per time step in seconds for different dx

# slots
nslots_dict = {} #set number of slots for each resolution
min_n_per_proc = 16 #25, minimum number of grid points per processor
even_split = False #1, force equal split between processors

#%%
'''Slot configurations for personal computer and cluster'''

dx_ind = [62.5, 125, 250] #resolutions which have their own job pools (if pooling is used)

if (("HOSTNAME" in os.environ) and (cluster_name in os.environ["HOSTNAME"])):
    cluster = True
    max_nslotsy = 2
    max_nslotsx = 7
    pool_size = 28 #number of cores per pool if job pooling is used
else:
    cluster = False
    max_nslotsy = 2
    max_nslotsx = 4

if not os.path.isdir(outpath):
    os.makedirs(outpath)


#%%

outpath_esc = outpath.replace("/", "\/") #need to escape slashes
if options.restart:
    init = False

start_time_dt = datetime.datetime.fromisoformat(start_time)
end_time_dt = datetime.datetime.fromisoformat(end_time)
start_d, start_t = start_time.split("_")
start_d = start_d.split("-")
start_t = start_t.split(":")
end_d, end_t = end_time.split("_")
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
    a["start_" + n] = di

if "dx" in param_grid_flat:
    dx = param_grid_flat["dx"]
else:
    dx = [a["dx"]]

#combine param grid and additional settings
combs = param_combs.copy()
for param, val in a.items():
    if param not in combs:
        combs[param] = val


#%%

for i in range(len(combs)):
    args = combs.loc[i].dropna().to_dict()
    param_comb = param_combs.loc[i]
    print("Config:")
    print("\n".join(str(param_comb).split("\n")[:-1]))

    r = args["dx"]
    dx_p.append(r)
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

    if use_gridpoints:
        args["e_we"] = max(math.ceil(args["lx"]/r), args["gridpoints"]) + 1
        lxr = (args["e_we"] -1)*r/args["lx"]
        if force_domain_multiple:
            if lxr != int(lxr):
                raise Exception("Domain size must be multiple of lx")
        args["e_sn"] = max(math.ceil(args["ly"]/r), args["gridpoints"]) + 1
    else:
        args["e_we"] = math.ceil(args["lx"]/r) + 1
        args["e_sn"] = math.ceil(args["ly"]/r) + 1

    vmem_init = max(vmem_init_min, int(vmem_init_per_grid_point*args["e_we"]*args["e_sn"]))

    if (nslots_dict is not None) and (r in nslots_dict) and (nslots_dict[r] is not None):
        nx, ny = nslots_dict[r]
    else:
        nx = misc.find_nproc(args["e_we"]-1, min_n_per_proc=min_n_per_proc, even_split=even_split )
        ny = misc.find_nproc(args["e_sn"]-1, min_n_per_proc=min_n_per_proc, even_split=even_split )
    if (nx == 1) and (ny == 1):
        nx = -1
        ny = -1
    nx = min(max_nslotsx, nx)
    ny = min(max_nslotsy, ny)

    nslotsi = nx*ny
    nslots.append(nslotsi)

    args["e_vert"] = args["nz"]
    eta, dz = vertical_grid.create_levels(nz=args["nz"], ztop=args["ztop"], method=args["dz_method"],
                                                      dz0=args["dz0"], plot=False, table=False)

    eta_levels = "'" + ",".join(eta.astype(str)) + "'"
    one_frame = False
    if r <= split_output_res: #split output in one timestep per file
        args["frames_per_outfile"] = 1
        for out_ind in output_streams.keys():
            if out_ind != 0:
                args["frames_per_auxhist{}".format(out_ind)] = 1
        one_frame = True

    for stream, (_, out_int) in output_streams.items():
        if stream > 0:
            args["auxhist{}_interval".format(stream)] = out_int
        else:
            args["history_interval_m"] = out_int

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
            args["km_opt"] = 4
            iofile = '"MESO_IO.txt"' #TODO: more general!
            pbl_scheme = args["bl_pbl_physics"]
        else:
            pbl_scheme = 0
            args["km_opt"] = 2
            args["sf_sfclay_physics"] = 1
            iofile = '"LES_IO.txt"'

        args["bl_pbl_physics"] = pbl_scheme

        if pbl_scheme in [5, 6]:
            args["bl_mynn_tkebudget"] = 1
        else:
            args["bl_mynn_tkebudget"] = 0
            args["bl_mynn_edmf"] = 0
            args["bl_mynn_edmf_tke"] = 0
            args["scalar_pblmix"] = 0

        #choose surface layer scheme that is compatible with PBL scheme
        if pbl_scheme in [1,2,3,4,5,7,10]:
            sfclay_scheme = pbl_scheme
        elif pbl_scheme == 6:
            sfclay_scheme = 5
        else:
            sfclay_scheme = 1

        args["sf_sfclay_physics"] = sfclay_scheme

        if "init_pert" not in args:
            if no_pert:
                args["init_pert"] = False
            elif all_pert or (r <= args["pert_res"]):
                args["init_pert"] = True
            else:
                args["init_pert"] = False

        # delete non-namelist parameters
        del_args =  ["dx", "nz", "dz0","dz_method", "gridpoints", "lx", "ly", "spec_hfx", "spec_sw","composite_name",
                    "pert_res", "input_sounding", "repi", "n_rep", "isotropic_res", "pbl_res", "dt"]
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
        args_str["init_pert"] = misc.bool_to_fort(args_str["init_pert"])
        args_str["const_sw"] = misc.bool_to_fort(args_str["const_sw"])
        args_str = str(args_str).replace("{","").replace("}","").replace("'","").replace(":","").replace(",","")
        args_str = args_str +  " iofields_filename " + iofile
        args_str += " eta_levels " + eta_levels

        if vmem_init > 25e3:
            init_queue = "bigmem.q"
        else:
            init_queue = "std.q"

        args["nslots"] = nslotsi

    if rt is not None:
        rtri = rt/3600 #runtime in hours
    else:
        rtri = runtime_per_step[r] * run_hours/dt * rt_buffer
    args["rtr"] = rtri
    args["nx"] = nx
    args["ny"] = ny
    for arg, val in args.items():
        combs.loc[i, arg] = val #not needed; just for completeness of dataframe

    if options.debug:
        wrf_dir_i = wrf_dir_pre + "_debug"
        slot_comm = ""
    elif nslotsi > 1:
        wrf_dir_i = wrf_dir_pre + "_mpi"
        slot_comm = "-pe openmpi-fillup {}".format(nslotsi)
    else:
        wrf_dir_i = wrf_dir_pre
        slot_comm = ""

    wrf_dir.append(wrf_dir_i)
    if options.pool_jobs:
        slot_comm = "-pe openmpi-{0}perhost {0}".format(pool_size)
        vmemi = vmem_pool
    else:
        vmemi = max(vmem_min, int(vmem_per_grid_point*args["e_we"]*args["e_sn"]))
    vmem.append(vmemi)

    #create output ID for current configuration
    IDi = param_comb.copy()
    for p in IDi.keys():
        if (p in composite_params) and (p != "composite_name"):
            del IDi[p]
    IDi = "_".join(IDi.values.astype(str))
    IDi =  runID + "_" + IDi

    repi = args["repi"]
    n_rep = args["n_rep"]
    for rep in range(repi, n_rep+repi): #repetion loop
        IDr = IDi + "_" + str(rep)
        IDs.append(IDr)

        if options.init:
            print("")

            hist_paths = r""
            for stream, (outfile, _) in output_streams.items():
                outname = r"{}{}_{}".format(outpath_esc, outfile, IDr)
                if one_frame:
                    outname += "_<date>"

                if stream == 0:
                    stream_arg = "history_outname"
                else:
                    stream_arg = "auxhist{}_outname".format(stream)

                hist_paths = r'''{} {} "{}"'''.format(hist_paths, stream_arg, outname)

            args_str = args_str + hist_paths
            comm_args =dict(wrfv=wrf_dir_i, ideal_case=ideal_case, input_sounding=args["input_sounding"],
                            sleep=rep, nx=nx, ny=ny, wrf_args=args_str, run_path=run_path, build_path=build_path,qsub=int(options.use_qsub))
            if options.use_qsub:
                comm_args_str = ",".join(["{}='{}'".format(p,v) for p,v in comm_args.items()])
                comm = r"qsub -q {} -l h_vmem={}M -m {} -N {} -v {} init_wrf.job".format(init_queue, vmem_init, options.mail, IDr, comm_args_str)
            else:
                d = {}
                for p, v in comm_args.items():
                    os.environ[p] = str(v)

                os.environ["JOB_NAME"] = IDr
                comm = "bash init_wrf.job '{}' ".format(args_str)

            print(comm)
            if not options.check_args:
                os.system(comm)
            IDs = []
            rtr = []
            vmem = []
            nslots = []
        else:
            if options.restart:
                wdir = "{}/WRF_{}/".format(run_path,IDr)
                rstfiles = glob.glob(wdir + "wrfrst*")
                if len(rstfiles) == 0:
                    raise RuntimeError("No restart files found!")
                rstf = sorted(rstfiles)[-1].split("/")[-1].split("_")[-2:]
                start_time_rst = datetime.datetime.fromisoformat("_".join(rstf))
                end_time_rst = datetime.datetime.fromisoformat(end_time)
                rst_date, rst_time = rstf
                rst_date = rst_date.split("-")
                rst_time = rst_time.split(":")
                run_h = int((end_time_rst - start_time_rst).total_seconds()/3600)
                if run_h == 0:
                    continue
                rtri = runtime_per_step[r] * run_h/dt * rt_buffer
                rst_opt = "restart .true. start_year {} start_month {} start_day {} start_hour {} start_minute {} start_second {} run_hours {}".format(*rst_date, *rst_time, run_h)
                os.makedirs("{}/bk/".format(outpath), exist_ok=True) #move previous output in backup directory
                bk_files = glob.glob("{}/bk/*{}".format(outpath, IDr))
                if len(bk_files) != 0:
                    raise FileExistsError("Backup files for restart already present. Double check!")
                os.system("mv {}/*{} {}/bk/".format(outpath, IDr, outpath))
                os.system("bash search_replace.sh {0}/namelist.input {0}/namelist.input {1}".format(wdir, rst_opt))
            rtr.append(rtri)

            split_res = False
            last_id = False
            dx_p_set = [str(int(rs)) for rs in set(dx_p)]
            if (np.in1d(dx_ind, dx_p).any()) and (len(dx_p_set) > 1):
                split_res = True
            elif (rep == n_rep+repi-1) and (i == len(combs)):
                last_id = True

            if (sum(nslots) >= pool_size) or (not options.pool_jobs) or split_res or last_id: #submit current pool of jobs
                print("")
                resched_i = False
                #if pool is already too large: cut out last job, which is rescheduled afterwards
                if (sum(nslots) > pool_size) or split_res:
                    nslots = nslots[:-1]
                    wrf_dir = wrf_dir[:-1]
                    vmem = vmem[:-1]
                    rtr = rtr[:-1]
                    IDs = IDs[:-1]
                    dx_p = dx_p[:-1]
                    resched_i = True
                    dx_p_set = [str(int(rs)) for rs in set(dx_p)]

                print("Submit IDs: {}".format(IDs))
                print("with total cores: {}".format(sum(nslots)))
                rtp = max(rtr)
                rtp = "{:02d}:{:02d}:00".format(math.floor(rtp), math.ceil((rtp - math.floor(rtp))*60))
                nslots = " ".join([str(ns) for ns in nslots])
                wrf_dir = " ".join([str(wd) for wd in wrf_dir])
                vmemp = int(sum(vmem)/len(vmem))
                if options.pool_jobs:
                    job_name = "pool_" + "_".join(IDs)
                else:
                    job_name = IDr
                jobs = " ".join(IDs)

                comm_args =dict(wrfv=wrf_dir, nslots=nslots,jobs=jobs, pool=int(options.pool_jobs), run_path=run_path, cluster=int(cluster))
                if options.use_qsub:
                    comm_args_str = ",".join(["{}='{}'".format(p,v) for p,v in comm_args.items()])
                    comm = r"qsub -q {} -N {} -l h_rt={} -l h_vmem={}M {} -m {} -v {} run_wrf.job".format(queue, job_name, rtp, vmemp, slot_comm, options.mail, comm_args_str)
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
                    nslots = [wrf_dir_i]
                    dx_p = [r]

                else:
                    IDs = []
                    rtr = []
                    vmem = []
                    nslots = []
                    wrf_dir = []
                    dx_p = []

                print(comm)
                if not options.check_args:
                    os.system(comm)

            else:
                pi += 1


