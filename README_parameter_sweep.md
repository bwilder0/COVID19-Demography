# Running parameter sweeps

1. Open `parameter_sweep.py`
2. At the top of the file, set the parameters to be whatever you need. That is edit the contents of these arrays
	
	```
	p_infect_given_contact_list = [round(0.02+0.001*i, 3) for i in range(12)]
	mortality_multiplier_list = [1, 2, 3, 4, 5]
	start_date_list = [10, 13, 16, 19, 22, 25]
	```

3. Run the following command:
`python3 parameter_sweep.py N_SIMS_PER_COMBO N_SIMS_PER_JOB --do_the_math`

   - Where:
       - N_SIMS_PER_COMBO = number of simulations you want per parameter combination
       - N_SIMS_PER_JOB = number of simulations you want to run per slurm job

   - Note: the larger N_SIMS_PER_JOB, the more time and memory it will take.
          But the smaller N_SIMS_PER_JOB, the less time and memory it will take, but more jobs and CPUs.

4. The script will print out the command you need to pass to slurm in order to run the simulations. e.g.,

	`python3 parameter_sweep.py 50 10 --do_the_math --sim_name lombardy_batch` 
	
	prints
	
	```
	NUM COMBOS: 360
	NUM INDICES: 1800

	Please launch jobs indexed from 0 to 1799, e.g.,
	sbatch --array=0-1799 job.parameter_sweep.sh 50 10 lombardy_batch
	```

5. On a slurm-capable device, run the above sbatch command. Note that results will be saved in the directory you specify as the final (4th) position argument of the sbatch command (so lombardy_batch in the above example)


6. When things finish, you can either jump to step 8 to plot immediately, or run step 7 to combine your result files into a more compact format (will make the rest of the plot code faster)

7. Open combine_results.py. In the first `if country == 'x'` block, replace the parameter lists with your own. Also replace `runs = ['lombardy_327_0', 'lombardy_327_1']` with the list of directories where you saved the results of all your previous simulations of this set of parameters (so the directories you specified in step 5.) In `NUM_JOBS`, specify how many jobs were run per combo (N_SIMS_PER_COMBO / N_SIMS_PER_JOB) per batch. Set `combined_dir = 'lombardy_327_combined'` or whatever you wish as an output directory. Run `python3 combine_results.py N_PROCESSES` where N_PROCESSES = number of processes you want (the code is parallelized).

8. To plot results, open `parse_parameter_sweep.py`

9. In the `if country == 'x'` block at the top (near line 18), replace the parameter lists with your own.

10. In the `if country == 'x'` block around line 78, replace `runs = []` with the list of directories containing your results. Note that it can handle many directories. If you ran many batches of trials, you can specify the directories containing the results of each here and the plotting code will retrieve them. Or if you ran step 7, you only need to pass the one combined directory (this will be faster.)

11. Run `python3 parse_parameter_sweep.py`
