import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import pickle
import json

#prex='sharegpt'
prex='govreport'
dir='../config/'

# load the distribution parameters from the pickle file
with open(dir+f'{prex}_distribution_params.pkl', 'rb') as f:
    params = pickle.load(f)
    input_params = params['input_params']
    output_params = params['output_params']

num_requests = 20000  # Number of simulated requests

# using gamma distribution to simulate token lengths
simulated_input_lengths = gamma.rvs(*input_params, size=num_requests)
simulated_output_lengths = gamma.rvs(*output_params, size=num_requests)

# ensure the lengths are at least 1 token
simulated_input_lengths = np.maximum(np.round(simulated_input_lengths).astype(int), 1).astype(int).tolist()
simulated_output_lengths = np.maximum(np.round(simulated_output_lengths).astype(int), 1).astype(int).tolist()

with open(dir+f'simulated_{prex}_input_lengths.json', 'w') as f:
    json.dump(simulated_input_lengths, f)

with open(dir+f'simulated_{prex}_output_lengths.json', 'w') as f:
    json.dump(simulated_output_lengths, f)