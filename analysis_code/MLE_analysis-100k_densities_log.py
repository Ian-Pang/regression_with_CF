#!/usr/bin/env python

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.rcParams['figure.dpi'] = 150

from nflows import transforms, distributions, flows
from nflows.utils import torchutils

import h5py
torch.set_default_dtype(torch.float64)

particle = 'piplus'

preproc = 'log10'

parser = argparse.ArgumentParser()


parser.add_argument('--weights_dir', default="/home/yp325/regression_project/regression_with_CF/results_variance/run_1/saved_checkpoints/",
                    help='Where to load weights from')

parser.add_argument('--results_dir', default="./data_uniform_densities/data_uniform_100k_run_1",
                    help='Where to store the results')

parser.add_argument('--which_cuda', default=0, type=int,
                    help='Which cuda device to use')

parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()

# folder containing the GEANT data used to train/validate Caloflow
source_folder = '/home/yp325/regression_project/data_discrete_100k'

weights_folder = args.weights_dir

results_folder = args.results_dir

# utilities, some are no longer used with log10 preproc
ALPHA = 1e-6
def logit(x):
    """ returns logit of input """
    return torch.log(x / (1.0 - x))

def logit_trafo(x):
    """ implements logit trafo of MAF paper https://arxiv.org/pdf/1705.07057.pdf """
    local_x = ALPHA + (1. - 2.*ALPHA) * x
    return logit(local_x)

def inverse_logit(x, clamp_low=0., clamp_high=1.):
    """ inverts logit_trafo(), clips result if needed """
    return ((torch.sigmoid(x) - ALPHA) / (1. - 2.*ALPHA)).clamp_(clamp_low, clamp_high)

def trafo_to_unit_space(energy_array):
    """ transforms energy array to be in [0, 1] """
    num_dim = len(energy_array[0])-2
    ret = [(torch.sum(energy_array[:, :-1], dim=1)/energy_array[:, -1]).unsqueeze(1)]
    for n in range(num_dim):
        ret.append((energy_array[:, n]/energy_array[:, n:-1].sum(dim=1)).unsqueeze(1))
    return torch.cat(ret, 1).clamp_(0., 1.)

def trafo_to_energy_space(unit_array, etot_array):
    """ transforms unit array to be back in energy space """
    assert len(unit_array) == len(etot_array)
    num_dim = len(unit_array[0])
    unit_array = torch.cat((unit_array, torch.ones(size=(len(unit_array), 1)).to(unit_array.device)), 1)
    ret = [torch.zeros_like(etot_array)]
    ehat_array = unit_array[:, 0] * etot_array
    for n in range(num_dim):
        ret.append(unit_array[:, n+1]*(ehat_array-torch.cat(ret).view(
            n+1, -1).transpose(0, 1).sum(dim=1)))
    ret.append(etot_array)
    return torch.cat(ret).view(num_dim+2, -1)[1:].transpose(0, 1)

class RandomPermutationLayer(transforms.Permutation):
    """ Permutes elements with random, but fixed permutation. Keeps pixel inside layer. """
    def __init__(self, features, dim=1):
        """ features: list of dimensionalities to be permuted"""
        assert isinstance(features, list), ("Input must be a list of integers!")
        permutations = []
        for index, features_entry in enumerate(features):
            current_perm = np.random.permutation(features_entry)
            if index == 0:
                permutations.extend(current_perm)
            else:
                permutations.extend(current_perm + np.cumsum(features)[index-1])
        super().__init__(torch.tensor(permutations), dim)

class InversionLayer(transforms.Permutation):
    """ Inverts the order of the elements in each layer.  Keeps pixel inside layer. """
    def __init__(self, features, dim=1):
        """ features: list of dimensionalities to be inverted"""
        assert isinstance(features, list), ("Input must be a list of integers!")
        permutations = []
        for index, features_entry in enumerate(features):
            current_perm = np.arange(features_entry)[::-1]
            if index == 0:
                permutations.extend(current_perm)
            else:
                permutations.extend(current_perm + np.cumsum(features)[index-1])
        super().__init__(torch.tensor(permutations), dim)

def add_noise(input_tensor,noiseamount=1e-9):
    noise = np.random.rand(*input_tensor.shape)*noiseamount
    return input_tensor+noise

# # Define and load models

# define Flow I
flow_params_Flow_I = {'num_blocks': 2, #num of layers per block
                      'features': 6,
                      'context_features': 1,
                      'hidden_features': 64,
                      'use_residual_blocks': False,
                      'use_batch_norm': False,
                      'dropout_probability': 0.,
                      'activation':F.relu,
                      'random_mask': False,
                      'num_bins': 8,
                      'tails':'linear',
                      'tail_bound': 14,
                      'min_bin_width': 1e-6,
                      'min_bin_height': 1e-6,
                      'min_derivative': 1e-6}
flow_i_blocks = []
for _ in range(6):
    flow_i_blocks.append(transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
        **flow_params_Flow_I))
    flow_i_blocks.append(transforms.RandomPermutation(6))
flow_i_transform = transforms.CompositeTransform(flow_i_blocks)
flow_i_base_distribution = distributions.StandardNormal(shape=[6])
flow_i = flows.Flow(transform=flow_i_transform, distribution=flow_i_base_distribution)

total_parameters = sum(p.numel() for p in flow_i.parameters() if p.requires_grad)

print("Flow I has {} parameters".format(int(total_parameters)))

# load trained weights
checkpoint = torch.load(weights_folder+'Flow_I/'+particle+'_log.pt', map_location='cpu')
flow_i.load_state_dict(checkpoint['model_state_dict'])
flow_i.eval() #this is super important - without this, the model still uses dropout!


# ## Maximum-likelihood inference

my_device = torch.device('cuda:'+str(args.which_cuda) \
                               if torch.cuda.is_available() and not args.no_cuda else 'cpu')
n_events = 100000
fixed_energies = np.linspace(10,90,9)

flow_i.to(my_device)

log_like = {}


for i in range(9):
    log_like[fixed_energies[i]]=[]

for jj in range(9):

    dataset = h5py.File(os.path.join(source_folder, '{}_'.format(particle))+str(int(fixed_energies[jj]))+'GeV.hdf5', 'r')
    considered_idx = np.arange(n_events)

    with torch.no_grad():
        
        x0 = add_noise(torch.tensor(dataset['layer_0'][considered_idx,:])/ 1e5)
        x1 = add_noise(torch.tensor(dataset['layer_1'][considered_idx,:])/ 1e5)
        x2 = add_noise(torch.tensor(dataset['layer_2'][considered_idx,:])/ 1e5)
        x3 = add_noise(torch.tensor(dataset['layer_3'][considered_idx,:])/ 1e5)
        x4 = add_noise(torch.tensor(dataset['layer_4'][considered_idx,:])/ 1e5)
        x5 = add_noise(torch.tensor(dataset['layer_5'][considered_idx,:])/ 1e5)
        E  = torch.tensor(dataset["energy"][considered_idx,:])/1e2

        E0 = x0.sum(dim=(1, 2))
        E1 = x1.sum(dim=(1, 2))
        E2 = x2.sum(dim=(1, 2))
        E3 = x3.sum(dim=(1, 2))
        E4 = x4.sum(dim=(1, 2))
        E5 = x5.sum(dim=(1, 2))

        E_scan = torch.linspace(fixed_energies[jj]*0.5,fixed_energies[jj]*2.0,1600)

        for i in range(n_events):
            if (i%1000==0):
                print("  ",jj,i/n_events)
            local_energy = E_scan
            local_mult = len(local_energy)

            local_E0 = E0[i].repeat(local_mult)
            local_E1 = E1[i].repeat(local_mult)
            local_E2 = E2[i].repeat(local_mult)
            local_E3 = E3[i].repeat(local_mult)
            local_E4 = E4[i].repeat(local_mult)
            local_E5 = E5[i].repeat(local_mult)

            if preproc == 'caloflow':
                x_flow1 = trafo_to_unit_space(torch.cat((local_E0, local_E1, local_E2, local_energy.unsqueeze(1)/100.), 1))
                x_flow1 = logit_trafo(x_flow1)
            elif preproc == 'log10':
                x_flow1 = torch.stack((local_E0, local_E1, local_E2, local_E3, local_E4, local_E5), 1)
                x_flow1 = 2.*(torch.log10((x_flow1*1e5)+1e-8)-1.).to(my_device)

            y_flow1 = torch.log10(local_energy/10.).unsqueeze(1).to(my_device)

            LL_1 = flow_i.log_prob(x_flow1, y_flow1).detach().cpu().numpy()

            log_like[fixed_energies[jj]].append(LL_1)

    np.save(results_folder+"/log_like_"+str(jj),log_like[fixed_energies[jj]])











