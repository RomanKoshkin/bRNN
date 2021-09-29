import time, sys, os, subprocess, pickle, shutil, itertools, string, warnings
import numpy as np
from scipy.stats import pearsonr
from cClasses import cClassOne
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


import networkx as nx
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph

from sknetwork.clustering import Louvain, modularity
from sknetwork.linalg import normalize
from sknetwork.utils import bipartite2undirected, membership_matrix

def clusterize(w):
    x = np.copy(w[:2500, :2500])
    G = nx.from_numpy_matrix(x)
    adj_mat = nx.to_numpy_array(G)
    louvain = Louvain()
    labels = louvain.fit_transform(adj_mat)
    mod = modularity(adj_mat, labels)

    labels_unique, counts = np.unique(labels, return_counts=True)

    tmp = sorted([(i, d) for i, d in enumerate(labels)], key=lambda tup:tup[1], reverse=True)
    newids = [i[0] for i in tmp]

    W_ = x
    W_ = W_[newids, :]
    W_ = W_[:, newids]
    return W_, labels, counts, mod


t = time.time()

NumCyclesAfterEquilibrium = 120
save_wts_at_ms = [5000, 400000, 700000, 1350000, 5350000, 13000000]
path_to_save_wts_on_bucket = '/bucket/FukaiU/Roman/slurm_big_HAGAwithFD/'
path_to_save_wts_on_flash  = '/flash/FukaiU/roman/GRID_withFD/'
slurm_out_folder = 'slurm_big'
df_fnanme = 'mod_big.txt'
T_ms = 14000000
stride_ms = 5000
dt = 0.01

weStimulate = False
set_evolved_weights = False
embed_assemblies = 0 # 0 if no assemblies
dump_weights = False

SLURM_ARRAY_JOB_ID = sys.argv[1]
SLURM_ARRAY_TASK_ID = sys.argv[2]
ARG = int(sys.argv[3])

U = 0.2
SET = 0

NE = 2500
NI = 500


df = pd.read_pickle('df.pkl')

# the order of keys IS IMPORTANT for the cClasses not to break down
params = {
    "alpha": 50.0,    # Degree of log-STDP (50.0)
    "usd": 0.1,       # Release probability of a synapse (0.05 - 0.5)
    "JEI": 0.15,      # 0.15 or 0.20

    "T": 1800*1000.0,   # simulation time, ms
    "h": 0.01,          # time step, ms ??????

    # probability of connection
    "cEE": 0.2, # 
    "cIE": 0.2, #
    "cEI": 0.5, #
    "cII": 0.5, #

    # Synaptic weights
    "JEE": 0.15, #
    "JEEinit": 0.16, # ?????????????
    "JIE": 0.15, # 
    "JII": 0.06, #
    
    #initial conditions of synaptic weights
    "JEEh": 0.15, # Standard synaptic weight E-E
    "sigJ": 0.3,  #

    "Jtmax": 0.25, # J_maxˆtot
    "Jtmin": 0.01, # J_minˆtot # ??? NOT IN THE PAPER

    # Thresholds of update
    "hE": 1.0, # Threshold of update of excitatory neurons
    "hI": 1.0, # Threshold of update of inhibotory neurons

    "IEex": 2.0, # Amplitude of steady external input to excitatory neurons
    "IIex": 0.5, # Amplitude of steady external input to inhibitory neurons
    "mex": 0.3,        # mean of external input
    "sigex": 0.1,      # variance of external input

    # Average intervals of update, ms
    "tmE": 5.0,  #t_Eud EXCITATORY
    "tmI": 2.5,  #t_Iud INHIBITORY
    
    #Short-Term Depression
    "trec": 600.0,     # recovery time constant (tau_sd, p.13 and p.12)
    "Jepsilon": 0.001, # ????????
    
    # Time constants of STDP decay
    "tpp": 20.0,  # tau_p
    "tpd": 40.0,  # tau_d
    "twnd": 500.0, # STDP window lenght, ms
    
    "g": 1.25,        # ??????
    
    #homeostatic
    "itauh": 100,       # decay time of homeostatic plasticity, (100s)
    "hsd": 0.1,
    "hh": 10.0,  # SOME MYSTERIOUS PARAMETER
    "Ip": 1.0, # External current applied to randomly chosen excitatory neurons
    "a": 0.20, # Fraction of neurons to which this external current is applied
    
    "xEinit": 0.02, # the probability that an excitatory neurons spikes at the beginning of the simulation
    "xIinit": 0.01, # the probability that an inhibitory neurons spikes at the beginning of the simulation
    "tinit": 100.00, # period of time after which STDP kicks in (100.0)
    "U": 0.6,
    "taustf": 200,
    "taustd": 500,
    "Cp": 0.01875,
    "Cd": 0.0075,
    "HAGA": True,
    "asym": True,
    "stimIntensity": 0.55,
} 


params['asym'] = 0
params['HAGA'] = 1   
params['JEE'] = 0.15

params["U"] = df.iloc[ARG]['U']
params['g'] = 2.5
params["tinit"] = 100
params["JEEinit"] = 0.16
params["Cp"] = df.iloc[ARG]['Cp']
params["Cd"] = df.iloc[ARG]['Cd']
params["tpp"] = df.iloc[ARG]['tpp']
params["tpd"] = df.iloc[ARG]['tpd']
params["taustf"] = df.iloc[ARG]['taustf']
params["taustd"] = df.iloc[ARG]['taustd']
params["alpha"] = 50.00
params["itauh"] = 100

m = cClassOne(NE, NI, ARG)
m.setParams(params)
ret = m.getState()

# check if the parameters have been set:
for var_name in params.keys():
    try:
        if params[var_name] != getattr(ret, var_name):
            print("{} doesn't match".format(var_name))
    except Exception as e: 
        print(e)


tt = time.time()
W0 = np.copy(m.getWeights())
TIMES = np.arange(0, T_ms, stride_ms)


saveWeightsAndSpikeStates = False # this will change to True once the clusters stabilize
NCLUST = []
lookback_strides = int(1200000 / stride_ms) # we'll be looking 12 strides (1 m back) to see if n_clust has stabilized
cyclesAfterEquilibrium = 0

for i in range(len(TIMES)):
    
    t = TIMES[i]
   
    ret = m.getState()
    W = m.getWeights()
    
    w_, labels, counts, mod = clusterize(W)
    n_clust_now = np.unique(labels).shape[0]
    NCLUST.append(n_clust_now) # keep track of the number of clusters every 5 s
    print(f'n_clust_now: {n_clust_now}')

    # #############################
    # if t > 4500:
    #     NCLUST = [8]*40
    # print(NCLUST)
    # print(lookback_strides)
    # print(NumCyclesAfterEquilibrium)
    # #############################3

    if not saveWeightsAndSpikeStates:
        print('Not saving weights and spike states yet.')
        if (mod > 0.1) & (len(NCLUST) > lookback_strides): # <!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!
            NCLUST_tail = NCLUST[-lookback_strides:] # has the number of clusters been the same over the lookback period?
            if all(el == NCLUST_tail[0] for el in NCLUST_tail):
                # if yes, we start saving weights and spike states
                saveWeightsAndSpikeStates = True
                m.saveSpikes(1) # 0 to disable saving
                print("Clusters have stabilized")
                print(f'Will run for {NumCyclesAfterEquilibrium} strides {NumCyclesAfterEquilibrium * stride_ms} ms and save spikes.')

    if saveWeightsAndSpikeStates:
        cyclesAfterEquilibrium += 1
        if cyclesAfterEquilibrium > NumCyclesAfterEquilibrium:
            print('SAVING weights and spike states.')

            ret = m.getState()
            W = m.getWeights()
            ys = m.getys()
            F = m.getF()
            D = m.getD()
        
            plt.imshow(w_, vmin=0.01, vmax=0.5)
            plt.title(f'mod: {mod:.2f} n_clust:{len(counts)}')
            # fname = f'{slurm_out_folder}/{ARG}_{t}_{mod}.png'
            fname = f'{slurm_out_folder}/{ARG}.png'
            plt.savefig(fname, dpi=300)
            command = f'scp {fname} roman-koshkin@deigo.oist.jp:{path_to_save_wts_on_bucket}'
            os.system(command)
            command = f'rm {fname}'
            os.system(command)

            # fname = f'wts_sorted_{ARG}_{t}_{mod}.pkl'
            fname = f'wts_sorted_{ARG}.pkl'
            with open(fname, 'wb') as f:
                pickle.dump(w_, f)
            command = f'scp {fname} roman-koshkin@deigo.oist.jp:{path_to_save_wts_on_bucket}'
            os.system(command)
            command = f'rm {fname}'
            os.system(command)

            # fname = f'wts_org_{ARG}_{t}_{mod}.pkl'
            fname = f'wts_org_{ARG}.pkl'
            with open(fname, 'wb') as f:
                pickle.dump(W, f)
            command = f'scp {fname} roman-koshkin@deigo.oist.jp:{path_to_save_wts_on_bucket}'
            os.system(command)
            command = f'rm {fname}'
            os.system(command)

            # fname = f'ys_{ARG}_{t}.pkl'
            fname = f'ys_{ARG}.pkl'
            with open(fname, 'wb') as f:
                pickle.dump(ys, f)
            command = f'scp {fname} roman-koshkin@deigo.oist.jp:{path_to_save_wts_on_bucket}'
            os.system(command)
            command = f'rm {fname}'
            os.system(command)

            # fname = f'D_{ARG}_{t}.pkl'
            fname = f'D_{ARG}.pkl'
            with open(fname, 'wb') as f:
                pickle.dump(D, f)
            command = f'scp {fname} roman-koshkin@deigo.oist.jp:{path_to_save_wts_on_bucket}'
            os.system(command)
            command = f'rm {fname}'
            os.system(command)

            # fname = f'F_{ARG}_{t}.pkl'
            fname = f'F_{ARG}.pkl'
            with open(fname, 'wb') as f:
                pickle.dump(F, f)
            command = f'scp {fname} roman-koshkin@deigo.oist.jp:{path_to_save_wts_on_bucket}'
            os.system(command)
            command = f'rm {fname}'
            os.system(command)

            m.dumpSpikeStates()

            fname = f'X_{ARG}'
            command = f'scp {fname} roman-koshkin@deigo.oist.jp:{path_to_save_wts_on_bucket}'
            os.system(command)
            command = f'rm {fname}'
            os.system(command)

            fname = f'DSPTS_{ARG}'
            command = f'scp {fname} roman-koshkin@deigo.oist.jp:{path_to_save_wts_on_bucket}'
            os.system(command)
            command = f'rm {fname}'
            os.system(command)

            fname = f'SPTS_{ARG}'
            command = f'scp {fname} roman-koshkin@deigo.oist.jp:{path_to_save_wts_on_bucket}'
            os.system(command)
            command = f'rm {fname}'
            os.system(command)

            fname = f'{path_to_save_wts_on_flash}spike_times_{ARG}'  # <!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!<!
            command = f'scp {fname} roman-koshkin@deigo.oist.jp:{path_to_save_wts_on_bucket}'
            os.system(command)
            command = f'rm {fname}'
            os.system(command)

            break

    print(f'Modularity: {mod}')

    m.sim(int(stride_ms//dt))
    print('Stride complete')

    with open(df_fnanme, 'a') as f:
        f.writelines(f'{SLURM_ARRAY_JOB_ID}, {SLURM_ARRAY_TASK_ID}, {ARG}, {t}, {mod}, {len(counts)}, {np.min(counts)}, {np.max(counts)}\n')     
    tt = time.time()
