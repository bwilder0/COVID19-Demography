import pandas as pd 
import numpy as np 
import os
import glob
from itertools import product
import datetime
import matplotlib.pyplot as plt
import sys
import multiprocessing

N_PROCESSES = int(sys.argv[1])
# combined_dir = sys.argv[1]


country = 'Italy'


N = None

combined_dir = ''
master_combo_list = None

# JUST HERE FOR REFERENCE
if country == 'Italy':
    N = 10000000.
    # p_infect_given_contact_list = [round(0.02+0.001*i, 3) for i in range(3)]
    # mortality_multiplier_list = [1, 2]
    # start_date_list = [19, 22, 25]

    p_infect_given_contact_list = [round(0.02+0.001*i, 3) for i in range(12)]
    mortality_multiplier_list = [1, 2, 3, 4, 5]
    start_date_list = [10, 13, 16, 19, 22, 25]

    master_combo_list = list(product(p_infect_given_contact_list, mortality_multiplier_list, start_date_list))
    N_COMBOS = len(master_combo_list)
    NUM_DATES = len(start_date_list)
    runs = ['lombardy_327_0', 'lombardy_327_1']
    NUM_JOBS = [25, 25] # how many jobs were in each run? N_SIMS_PER_COMBO / N_SIMS_PER_JOB

    combined_dir = 'lombardy_327_combined_3'


elif country == 'China':
    p_infect_given_contact_list = [0.018, 0.019, 0.0195, 0.02, 0.021, 0.022, 0.023]
    mortality_multiplier_list = [1]
    start_date_list = [8, 10, 12, 13, 15, 17, 19]

    master_combo_list = list(product(p_infect_given_contact_list, mortality_multiplier_list, start_date_list))
    N_COMBOS = len(master_combo_list)
    NUM_DATES = len(start_date_list)
    
    runs = [
        'hubei_distributed_independent_0',
#        'lombardy_distributed_independent_1'
        # 'lombardy_distributed_independent_2'
    ]



name_templates = []


for run_name in runs:
    dirname = os.path.join('parameter_sweeps',run_name)
    name_template = run_name+'_paramsweep_n%s_i%s_N%s_p%s_m%s_s%s_%s.csv'
    name_templates.append(os.path.join(dirname, name_template))

dirname = os.path.join('parameter_sweeps',combined_dir)
if not os.path.exists(dirname):
    os.mkdir(dirname)

out_template = combined_dir+'_paramsweep_n%s_i%s_N0_p%s_m%s_s%s_%s.csv'
out_template = os.path.join(dirname, out_template)


def combine_data(datatype, i, name_templates):

    this_combo = master_combo_list[i]

    pigc_val = this_combo[0]
    mm_val = this_combo[1]
    sd_val = this_combo[2]

    trial_data = []
    for ind,name_template in enumerate(name_templates):
        for trial_num in range(NUM_JOBS[ind]):
            fname = name_template%(N, i, trial_num, pigc_val, mm_val, sd_val, datatype)

            try:
                batch_data = pd.read_csv(fname,header=None,sep='\t').values
            except FileNotFoundError:
                continue


            if batch_data.shape[0]==0:
                batch_data = np.array([[0]])

            # if num_d == 3:
                # batch_data = batch_data.reshape(batch_data.shape[0], num_ages, -1)

            trial_data.append(batch_data)
    trial_data = np.concatenate(trial_data,axis=0)

    outname = out_template%(N, i, pigc_val, mm_val, sd_val, datatype)

    df = pd.DataFrame(trial_data)
    df.to_csv(outname,sep='\t',index=False,header=False, na_rep='NA')


datatypes_n_1 = [
        'r0_tot', 
        'mse'
]

datatypes_n_t = [
        'susceptible', 
        'exposed', 
        'deaths', 
        'mild', 
        'severe', 
        'critical', 
        'recovered', 
        'quarantine', 
        'r0_time',
        'cfr_time', 
        'frac_over_70_time', 
        'frac_below_30_time', 
        'median_age_time'
    
]

# These will be the biggest storage burden. Cut where you can.
datatypes_n_a_t = [
        'infected_age_time', 
        'total_age_time',
        'cfr_age_time',
        'dead_age_time'
]



def worker(args):

    i = args[0]
    print("worker",i)
    for d in datatypes_n_1:
        trial_data = combine_data(d,i,name_templates)

    for d in datatypes_n_t:
        trial_data = combine_data(d,i,name_templates)

    for d in datatypes_n_a_t:
        trial_data = combine_data(d,i,name_templates)


worker_args = [[i] for i in range(N_COMBOS)]
    
pool = multiprocessing.Pool(processes=N_PROCESSES)
pool.map_async(worker, worker_args).get(9999999)
# list(map(worker, worker_args))


