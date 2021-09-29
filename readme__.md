really
launches grid search on compute/short
on compute the limit is many hours (2000 cores)
on short the limit is 2 hours (4000 cores)

to not exceed the maximum core count running at the same time, use `make_grid.ipynb`


Checklist

- compile with
	g++ -std=gnu++11 -Ofast -shared -fPIC -ftree-vectorize -march=native -mavx bmm_9_haga_grid.cpp -o ./bmm.dylib
	DON'T COMPILE ON THE LOGIN NODE, BUT WHEN YOU ARE IN A SRUN SESSION. THAT WILL ENSURE THAT THE BINARY IS COMPILED FOR THE MACHINE IT WILL ACTUALLY RUN ON, NOT ON THE LOGIN NODE. 
- set the needed parameters at the top of the grid_cell.py
	save_wts_at_ms = [5000, 400000, 700000, 1350000, 5350000, 14000000]
	path_to_save_wts_on_bucket = '/bucket/FukaiU/Roman/slurm_big/'
	slurm_out_folder = 'slurm_big'
	df_fnanme = 'mod_big.txt'
	T_ms = 14000000
- make new folders for 
	path_to_save_wts_on_bucket
	slurm_out_folder
- set delay_between_jobs_s based on the script in make_grid.ipynb
- set the --time=2-6:45:0 to allow enough time for a job to complete
- set the --output=./slurm_big/%j.out
- set the partition (either compute or short)






