# pylint: disable=invalid-name
""" Main script to run CaloFlow.
    code based on https://github.com/bayesiains/nflows and https://arxiv.org/pdf/1906.04032.pdf

    This is the cleaned-up version of the code. It supports running in a single mode:

    - use one flow to learn p(E_i|E_tot) and then train a single flow on normalized samples
      of all layers

    This code was used for the following publications:

    "CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows"
    by Claudius Krause and David Shih
    arxiv:2106.05285, Phys.Rev.D 107 (2023) 11, 113003

    "CaloFlow II: Even Faster and Still Accurate Generation of Calorimeter Showers with
     Normalizing Flows"
    by Claudius Krause and David Shih
    arXiv:2110.11377, Phys.Rev.D 107 (2023) 11, 113004

"""

######################################   Imports   #################################################
import argparse
import os
import time

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from nflows import transforms, distributions, flows
from nflows.utils import torchutils

from data import get_dataloader
from data import save_samples_to_file

torch.set_default_dtype(torch.float64)

#####################################   Parser setup   #############################################
parser = argparse.ArgumentParser()

# usage modes
parser.add_argument('--train', action='store_true', help='train a flow')
parser.add_argument('--generate', action='store_true', help='generate from a trained flow and plot')
parser.add_argument('--analyze', action='store_true', help='perform MLE analysis')
parser.add_argument('--evaluate', action='store_true', help='evaluate LL of a trained flow')
parser.add_argument('--evaluate_KL', action='store_true',
                    help='evaluate KL of a trained student flow')
parser.add_argument('--generate_to_file', action='store_true',
                    help='generate from a trained flow and save to file')
parser.add_argument('--save_only_weights', action='store_true',
                    help='Loads full model file (incl. optimizer) and saves only weights')
parser.add_argument('--save_every_epoch', action='store_true',
                    help='Saves weights (no optimizer) of every student epoch')

parser.add_argument('--student_mode', action='store_true',
                    help='Work with IAF-student instead of MAF-teacher')

parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--which_cuda', default=0, type=int,
                    help='Which cuda device to use')

parser.add_argument('--output_dir', default='./results', help='Where to store the output')
parser.add_argument('--results_file', default='results.txt',
                    help='Filename where to store settings and test results.')
parser.add_argument('--flowI_restore_file', type=str, default=None,
                    help='Flow I model file to restore.')
parser.add_argument('--restore_file', type=str, default=None,
                    help='Flow II teacher model file to restore.')
parser.add_argument('--student_restore_file', type=str, default=None,
                    help='Flow II student model file to restore.')
parser.add_argument('--data_dir', help='Where to find the training dataset')

# CALO specific
parser.add_argument('--with_noise', action='store_true', default=True,
                    help='Add 1e-8 noise (w.r.t. 100 GeV) to dataset to avoid voxel with 0 energy')
parser.add_argument('--particle_type', '-p', choices=['gamma', 'eplus', 'piplus', 'piplus_high', 'piplus_log'],
                    help='Which particle to shower, "gamma", "eplus", or "piplus"')
parser.add_argument('--threshold', type=float, default=0.01,
                    help='Threshold in MeV below which voxel energies are set to 0. in plots.')
parser.add_argument('--log_preprocess', action='store_true', default=False,
                    help='Do not go to u-space, instead use simple log10 preprocessing')

# MAF parameters
parser.add_argument('--n_blocks', type=int, default=8,
                    help='Total number of blocks to stack in a model (MADE in MAF).')
parser.add_argument('--student_n_blocks', type=int, default=8,
                    help='Total number of blocks to stack in the student model (MADE in IAF).')
parser.add_argument('--hidden_size', type=int, default=512,
                    help='Hidden layer size for each MADE block in an MAF.')
parser.add_argument('--student_hidden_size', type=int, default=504,
                    help='Hidden layer size for each MADE block in the student IAF.')
parser.add_argument('--student_width', type=float, default=1.,
                    help='Width of the base dist. that is used for student training.')
parser.add_argument('--n_hidden', type=int, default=1,
                    help='Number of hidden layers in each MADE.')
parser.add_argument('--activation_fn', type=str, default='relu',
                    help='What activation function of torch.nn.functional to use in the MADEs.')
parser.add_argument('--n_bins', type=int, default=8,
                    help='Number of bins if piecewise transforms are used')
parser.add_argument('--dropout_probability', '-d', type=float, default=0.05,
                    help='dropout probability')
parser.add_argument('--tail_bound', type=float, default=14., help='Domain of the RQS')
parser.add_argument('--beta', type=float, default=0.5,
                    help='Sets the relative weight between z-chi2 loss (beta=0) and x-chi2 loss')

# training params
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-4, help='Initial Learning rate.')
parser.add_argument('--log_interval', type=int, default=175,
                    help='How often to show loss statistics.')
parser.add_argument('--fully_guided', action='store_true', default=True,
                    help='Train student "fully-guided", ie enable train_xz and train_p')
parser.add_argument('--train_xz', action='store_true', default=False,
                    help="Train student with MSE of all intermediate x's and z's")
parser.add_argument('--train_p', action='store_true', default=False,
                    help='Train student with MSE of all MADE-NN outputs '+\
                    '(to-be parameters of the RQS)')

#######################################   helper functions   #######################################

# used in transformation between energy and logit space:
# (should match the ALPHA in data.py)
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

class IAFRQS(transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform):
    """ IAF version of nflows MAF-RQS"""
    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)
    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

class GuidedCompositeTransform(transforms.CompositeTransform):
    """Composes several transforms into one (in the order they are given),
       optionally returns intermediate results (steps) and NN outputs (p)"""

    def __init__(self, transforms):
        """Constructor.
        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__(transforms)
        self._transforms = torch.nn.ModuleList(transforms)

    @staticmethod
    def _cascade(inputs, funcs, context, direction, return_steps=False, return_p=False):
        steps = [inputs]
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(batch_size)
        ret_p = []
        for func in funcs:
            if hasattr(func.__self__, '_transform') and return_p:
                # in student IAF
                if direction == 'forward':
                    outputs, logabsdet = func(outputs, context)
                    ret_p.append(func.__self__._transform.autoregressive_net(outputs, context))
                else:
                    ret_p.append(func.__self__._transform.autoregressive_net(outputs, context))
                    outputs, logabsdet = func(outputs, context)
            elif hasattr(func.__self__, 'autoregressive_net') and return_p:
                # in teacher MAF
                if direction == 'forward':
                    ret_p.append(func.__self__.autoregressive_net(outputs, context))
                    outputs, logabsdet = func(outputs, context)
                else:
                    outputs, logabsdet = func(outputs, context)
                    ret_p.append(func.__self__.autoregressive_net(outputs, context))
            else:
                outputs, logabsdet = func(outputs, context)
            steps.append(outputs)
            total_logabsdet += logabsdet
        if return_steps and return_p:
            return outputs, total_logabsdet, steps, ret_p
        elif return_steps:
            return outputs, total_logabsdet, steps
        elif return_p:
            return outputs, total_logabsdet, ret_p
        else:
            return outputs, total_logabsdet

    def forward(self, inputs, context=None, return_steps=False, return_p=False):
        #funcs = self._transforms
        funcs = (transform.forward for transform in self._transforms)
        return self._cascade(inputs, funcs, context, direction='forward',
                             return_steps=return_steps, return_p=return_p)

    def inverse(self, inputs, context=None, return_steps=False, return_p=False):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs, context, direction='inverse',
                             return_steps=return_steps, return_p=return_p)

# TO-DO: check if needed:
def remove_nans(tensor):
    """removes elements in the given batch that contain nans
       returns the new tensor and the number of removed elements"""
    tensor_flat = tensor.flatten(start_dim=1)
    good_entries = torch.all(tensor_flat == tensor_flat, axis=1)
    res_flat = tensor_flat[good_entries]
    tensor_shape = list(tensor.size())
    tensor_shape[0] = -1
    res = res_flat.reshape(tensor_shape)
    return res, len(tensor) - len(res)

@torch.no_grad()
def logabsdet_of_base(noise, width=1.):
    """ for computing KL of student"""
    shape = noise.size()[1]
    ret = -0.5 * torchutils.sum_except_batch((noise/width) ** 2, num_batch_dims=1)
    log_z = torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi), dtype=torch.float64)
    return ret - log_z

def transform_to_energy(sample, arg, scaling):
    """ transforms samples from logit space to energy space, possibly applying a scaling factor
    """
    sample = ((torch.sigmoid(sample) - ALPHA) / (1. - 2.*ALPHA))

    sample0, sample1, sample2 = torch.split(sample, arg.dim_split, dim=-1)
    sample0 = (sample0 / sample0.abs().sum(dim=(-1), keepdims=True))\
        * scaling[:, 0].reshape(-1, 1, 1)
    sample1 = (sample1 / sample1.abs().sum(dim=(-1), keepdims=True))\
        * scaling[:, 1].reshape(-1, 1, 1)
    sample2 = (sample2 / sample2.abs().sum(dim=(-1), keepdims=True))\
        * scaling[:, 2].reshape(-1, 1, 1)
    sample3 = (sample3 / sample3.abs().sum(dim=(-1), keepdims=True))\
        * scaling[:, 3].reshape(-1, 1, 1)
    sample4 = (sample4 / sample4.abs().sum(dim=(-1), keepdims=True))\
        * scaling[:, 4].reshape(-1, 1, 1)
    sample5 = (sample5 / sample5.abs().sum(dim=(-1), keepdims=True))\
        * scaling[:, 5].reshape(-1, 1, 1)
    sample = torch.cat((sample0, sample1, sample2, sample3, sample4, sample5), 2)
    sample = sample*1e5
    return sample



def split_and_concat(generate_fun, batch_size, model, arg, num_pts, energies, rec_model):
    """ generates events in batches of size batch_size, if needed """
    starting_time = time.time()
    energy_split = energies.split(batch_size)
    ret = []
    for iteration, energy_entry in enumerate(energy_split):
        ret.append(generate_fun(model, arg, num_pts, energy_entry, rec_model).to('cpu'))
        print("Generated {}%".format((iteration+1.)*100. / len(energy_split)), end='\r')
    ending_time = time.time()
    total_time = ending_time - starting_time
    time_string = "Needed {:d} min and {:.1f} s to generate {} events in {} batch(es)."+\
        " This means {:.2f} ms per event."
    print(time_string.format(int(total_time//60), total_time%60, num_pts*len(energies),
                             len(energy_split), total_time*1e3 / (num_pts*len(energies))))
    print(time_string.format(int(total_time//60), total_time%60, num_pts*len(energies),
                             len(energy_split), total_time*1e3 / (num_pts*len(energies))),
          file=open(arg.results_file, 'a'))
    return torch.cat(ret)


@torch.no_grad()
def generate_to_file(model, arg, rec_model, num_events=100000, energies=None):
    """ generates samples from the trained model and saves them to file """
    if energies is None:
        energies = 0.99*torch.rand((num_events,)) + 0.01
    scaling = torch.reshape(energies, (-1, 1, 1)).to(arg.device)

    # adjust line below for smaller generation batch size or more than 1 sample per energy
    samples = split_and_concat(generate_single_with_rec, 10000, model, arg, 1, energies,
                               rec_model)
    filename = os.path.join(arg.output_dir, 'CaloFlow_'+arg.particle_type+'.hdf5')
    save_samples_to_file(samples, energies, filename, arg.threshold)

def train_and_evaluate(model, train_loader, test_loader, optimizer, arg, rec_model):
    """ As the name says, train the flow and evaluate along the way """
    best_eval_logprob = float('-inf')
    milestones = [50,150]
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                       milestones=milestones,
                                                       gamma=0.5,
                                                       verbose=True)
    for i in range(arg.n_epochs):
        train(model, train_loader, optimizer, i, arg)
        with torch.no_grad():
            eval_logprob, _ = evaluate(model, test_loader, i, arg)
            arg.test_loss.append(-eval_logprob.to('cpu').numpy())
        if eval_logprob > best_eval_logprob:
            best_eval_logprob = eval_logprob
            save_all(model, optimizer, arg)

        lr_schedule.step()

def save_all(model, optimizer, arg, is_student=False):
    """ saves the model and the optimizer of Flow II to file """
    if is_student:
        file_name = os.path.join(arg.output_dir,
                                 f"saved_checkpoints/Flow_II/{arg.particle_type}_full_student.pt")
    else:
        file_name = os.path.join(arg.output_dir,
                                 f"saved_checkpoints/Flow_II/{arg.particle_type}_full_sep19.pt")
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, file_name)
    #          os.path.join(arg.output_dir,
    #                       'student.pt' if is_student else 'model_checkpoint.pt'))

def save_weights(model, arg, is_student=False, name=None):
    """ saves the model of Flow II to file """
    if name is not None:
        file_name = name
    else:
        #file_name = 'student_weights.pt' if is_student else 'weight_checkpoint.pt'
        if is_student:
            file_name = os.path.join(arg.output_dir,
                                     f"saved_checkpoints/Flow_II/{arg.particle_type}_student.pt")
        else:
            file_name = os.path.join(arg.output_dir,
                                     f"saved_checkpoints/Flow_II/{arg.particle_type}.pt")
    torch.save({'model_state_dict': model.state_dict()},
               os.path.join(arg.output_dir, file_name))

def load_all(model, optimizer, arg, is_student=False):
    """ loads the model and optimizer for Flow II from file """
    if is_student:
        filename = arg.student_restore_file if arg.student_restore_file is not None\
            else os.path.join(arg.output_dir,
                              f"saved_checkpoints/Flow_II/{arg.particle_type}_full_student.pt")
            #else 'student.pt'
    else:
        filename = arg.restore_file if arg.restore_file is not None else\
            os.path.join(arg.output_dir,
                         f"saved_checkpoints/Flow_II/{arg.particle_type}_full.pt")
        #'model_checkpoint.pt'
    #checkpoint = torch.load(os.path.join(arg.output_dir, filename))
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(arg.device)
    model.eval()
    print(f"loaded model weights and optimizer from {filename}.")

def load_weights(model, arg, is_student=False):
    """ loads the model for Flow II from file """
    if is_student:
        filename = arg.student_restore_file if arg.student_restore_file is not None\
            else os.path.join(arg.output_dir,
                              f"saved_checkpoints/Flow_II/{arg.particle_type}_student.pt")
            #else 'student_weights.pt'
    else:
        filename = arg.restore_file if arg.restore_file is not None else\
            os.path.join(arg.output_dir, f"saved_checkpoints/Flow_II/{arg.particle_type}.pt")
        #'weight_checkpoint.pt'
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(arg.device)
    model.eval()
    print(f"loaded model weights from {filename}.")

def save_rec_flow(rec_model, arg):
    """saves flow that learns energies recursively (Flow I)"""
    torch.save({'model_state_dict': rec_model.state_dict()},
               os.path.join(arg.output_dir, f"saved_checkpoints/Flow_I/{arg.particle_type}_sep19.pt"))
               #os.path.join(arg.output_dir, arg.particle_type+'_rec.pt'))
               #os.path.join('./rec_energy_flow/', arg.particle_type+'.pt'))

def load_rec_flow(rec_model, arg):
    """ loads flow that learns energies recursively (Flow I)"""
    #checkpoint = torch.load(os.path.join('./rec_energy_flow/', arg.particle_type+'.pt'),
    #checkpoint = torch.load(os.path.join(arg.output_dir, arg.particle_type+'_rec.pt'),
    if arg.flowI_restore_file is None:
        filename = os.path.join(arg.output_dir, f"saved_checkpoints/Flow_I/{arg.particle_type}_sep19.pt")
    else:
        filename = arg.flowI_restore_file
    checkpoint = torch.load(filename, map_location='cpu')
    rec_model.load_state_dict(checkpoint['model_state_dict'])
    rec_model.to(arg.device)
    rec_model.eval()

def trafo_to_unit_space(energy_array):
    """ transforms energy array to be in [0, 1] """
    num_dim = len(energy_array[0])-2
    ret = [(torch.sum(energy_array[:, :-1], dim=1)/(4.65*energy_array[:, -1])).unsqueeze(1)]
    for n in range(num_dim):
        ret.append((energy_array[:, n]/energy_array[:, n:-1].sum(dim=1)).unsqueeze(1))
    return torch.cat(ret, 1).clamp_(0., 1.)

def trafo_to_energy_space(unit_array, etot_array):
    """ transforms unit array to be back in energy space """
    assert len(unit_array) == len(etot_array)
    num_dim = len(unit_array[0])
    unit_array = torch.cat((unit_array, torch.ones(size=(len(unit_array), 1))), 1)
    ret = [torch.zeros_like(etot_array)]
    ehat_array = unit_array[:, 0] * 4.65* etot_array
    for n in range(num_dim):
        ret.append(unit_array[:, n+1]*(ehat_array-torch.cat(ret).view(
            n+1, -1).transpose(0, 1).sum(dim=1)))
    ret.append(etot_array)
    return torch.cat(ret).view(num_dim+2, -1)[1:].transpose(0, 1)

################################# auxilliary NNs and classes #######################################

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
        for idx, features_entry in enumerate(features):
            current_perm = np.arange(features_entry)[::-1]
            if idx == 0:
                permutations.extend(current_perm)
            else:
                permutations.extend(current_perm + np.cumsum(features)[idx-1])
        super().__init__(torch.tensor(permutations), dim)


################## train and evaluation functions for recursive layer single flow ##################

def train(model, dataloader, optimizer, epoch, arg):
    """ train the flow one epoch """
    model.train()
    for i, data in enumerate(dataloader):
        x0 = data['layer_0']
        x1 = data['layer_1']
        x2 = data['layer_2']
        x3 = data['layer_3']
        x4 = data['layer_4']
        x5 = data['layer_5']
        E0 = data['layer_0_E']
        E1 = data['layer_1_E']
        E2 = data['layer_2_E']
        E3 = data['layer_3_E']
        E4 = data['layer_4_E']
        E5 = data['layer_5_E']
        E  = data['energy']

        #energy_dists = trafo_to_unit_space(torch.cat((E0.unsqueeze(1),
        #                                              E1.unsqueeze(1),
        #                                              E2.unsqueeze(1),
        #                                              E), 1))
        energy = torch.log10(E*10.)
        E0 = torch.log10(E0.unsqueeze(-1)+1e-8) + 2.
        E1 = torch.log10(E1.unsqueeze(-1)+1e-8) + 2.
        E2 = torch.log10(E2.unsqueeze(-1)+1e-8) + 2.
        E3 = torch.log10(E3.unsqueeze(-1)+1e-8) + 2.
        E4 = torch.log10(E4.unsqueeze(-1)+1e-8) + 2.
        E5 = torch.log10(E5.unsqueeze(-1)+1e-8) + 2.

        y = torch.cat((energy, E0, E1, E2, E3, E4, E5), 1).to(arg.device)

        layer0 = x0.view(x0.shape[0], -1).to(arg.device)
        layer1 = x1.view(x1.shape[0], -1).to(arg.device)
        layer2 = x2.view(x2.shape[0], -1).to(arg.device)
        layer3 = x3.view(x3.shape[0], -1).to(arg.device)
        layer4 = x4.view(x4.shape[0], -1).to(arg.device)
        layer5 = x5.view(x5.shape[0], -1).to(arg.device)
        x = torch.cat((layer0, layer1, layer2, layer3, layer4, layer5), 1)

        loss = - model.log_prob(x, y).mean(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        arg.train_loss.append(loss.tolist())

        if i % arg.log_interval == 0:
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch+1, arg.n_epochs, i, len(dataloader), loss.item()))
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch+1, arg.n_epochs, i, len(dataloader), loss.item()),
                  file=open(arg.results_file, 'a'))

@torch.no_grad()
def evaluate(model, dataloader, epoch, arg, num_batches=None):
    """Evaluate the model, i.e find the mean log_prob of the test set
       Energy is taken to be the energy of the image, so no
       marginalization is performed.
    """
    model.eval()
    loglike = []
    for batch_id, data in enumerate(dataloader):
        x0 = data['layer_0']
        x1 = data['layer_1']
        x2 = data['layer_2']
        x3 = data['layer_3']
        x4 = data['layer_4']
        x5 = data['layer_5']
        E0 = data['layer_0_E']
        E1 = data['layer_1_E']
        E2 = data['layer_2_E']
        E3 = data['layer_3_E']
        E4 = data['layer_4_E']
        E5 = data['layer_5_E']
        E  = data['energy']

        #energy_dists = trafo_to_unit_space(torch.cat((E0.unsqueeze(1),
        #                                              E1.unsqueeze(1),
        #                                              E2.unsqueeze(1),
        #                                              E), 1))
        energy = torch.log10(E*10.)
        E0 = torch.log10(E0.unsqueeze(-1)+1e-8) + 2.
        E1 = torch.log10(E1.unsqueeze(-1)+1e-8) + 2.
        E2 = torch.log10(E2.unsqueeze(-1)+1e-8) + 2.
        E3 = torch.log10(E3.unsqueeze(-1)+1e-8) + 2.
        E4 = torch.log10(E4.unsqueeze(-1)+1e-8) + 2.
        E5 = torch.log10(E5.unsqueeze(-1)+1e-8) + 2.

        y = torch.cat((energy, E0, E1, E2, E3, E4, E5), 1).to(arg.device)

        layer0 = x0.view(x0.shape[0], -1).to(arg.device)
        layer1 = x1.view(x1.shape[0], -1).to(arg.device)
        layer2 = x2.view(x2.shape[0], -1).to(arg.device)
        layer3 = x3.view(x3.shape[0], -1).to(arg.device)
        layer4 = x4.view(x4.shape[0], -1).to(arg.device)
        layer5 = x5.view(x5.shape[0], -1).to(arg.device)
        x = torch.cat((layer0, layer1, layer2, layer3, layer4, layer5), 1).to(arg.device)

        loglike.append(model.log_prob(x, y))
        if num_batches is not None:
            if batch_id == num_batches-1:
                break

    logprobs = torch.cat(loglike, dim=0).to(arg.device)

    logprob_mean = logprobs.mean(0)
    logprob_std = logprobs.var(0).sqrt()# / np.sqrt(len(dataloader.dataset))

    output = 'Evaluate ' + (epoch is not None)*'(epoch {}) -- '.format(epoch+1) +\
        'logp(x, at E(x)) = {:.3f} +/- {:.3f}'

    print(output.format(logprob_mean, logprob_std))
    print(output.format(logprob_mean, logprob_std), file=open(arg.results_file, 'a'))
    return logprob_mean, logprob_std

@torch.no_grad()
def generate_single_with_rec(model, arg, num_pts, energies, rec_model):
    """ Generate Samples from single flow with energy flow """
    model.eval()

    energy_dist_unit = sample_rec_flow(rec_model, num_pts, arg, energies).to('cpu')
    if arg.log_preprocess:
        energy_dist = (10**(0.5*energy_dist_unit + 1) - 1e-8)/1e5
    else:
        energy_dist = trafo_to_energy_space(energy_dist_unit, energies)
    energies = torch.log10(energies*10.).unsqueeze(-1)
    E0 = torch.log10(energy_dist[:, 0].unsqueeze(-1)+1e-8) + 2.
    E1 = torch.log10(energy_dist[:, 1].unsqueeze(-1)+1e-8) + 2.
    E2 = torch.log10(energy_dist[:, 2].unsqueeze(-1)+1e-8) + 2.
    E3 = torch.log10(energy_dist[:, 3].unsqueeze(-1)+1e-8) + 2.
    E4 = torch.log10(energy_dist[:, 4].unsqueeze(-1)+1e-8) + 2.
    E5 = torch.log10(energy_dist[:, 5].unsqueeze(-1)+1e-8) + 2.

    y = torch.cat((energies, E0, E1, E2, E3, E4, E5), 1).to(arg.device)

    samples = model.sample(num_pts, y)

    samples = transform_to_energy(samples, arg, scaling=energy_dist.to(arg.device))

    return samples

@torch.no_grad()
def generate_layerE(model, arg, num_pts, energies, rec_model):
    """ Generate Samples of layer E """
    model.eval()

    energy_dist_unit = sample_rec_flow(rec_model, num_pts, arg, energies).to('cpu')
    if arg.log_preprocess:
        energy_dist = (10**(0.5*energy_dist_unit + 1) - 1e-8)/1e3
    else:
        energy_dist = trafo_to_energy_space(energy_dist_unit, energies)
    return energy_dist

def MLE_analysis(model, arg, rec_model):
    model.eval()
    rec_model.eval()
    analysis_dataloader = get_dataloader(args.particle_type,
                                        args.data_dir,
                                        full=True,
                                        apply_logit=True,
                                        device=args.device,
                                        batch_size=1,
                                        with_noise = True,
                                        normed=False,
                                        normed_layer=True)

    e_inc = torch.linspace(0.01,1,1000)
    e_inc = torch.log10(e_inc*10.).unsqueeze(-1)
    for batch_id, data in enumerate(analysis_dataloader):
        x0 = data['layer_0']
        x1 = data['layer_1']
        x2 = data['layer_2']
        x3 = data['layer_3']
        x4 = data['layer_4']
        x5 = data['layer_5']
        E0 = data['layer_0_E']
        E1 = data['layer_1_E']
        E2 = data['layer_2_E']
        E3 = data['layer_3_E']
        E4 = data['layer_4_E']
        E5 = data['layer_5_E']

        true_E  = data['energy'] # this is the true incident energy from the dataset

        #print(E0.shape, E1.shape, E2.shape, E3.shape, E4.shape, E5.shape, e_inc[0].unsqueeze(0).shape)
        #print(E0, E1, E2, E3, E4, E5, e_inc[0].unsqueeze(0))
        results = true_E.squeeze(0).to(arg.device)
        print(E0,E1,E2,E3,E4,E5,true_E)
        x = trafo_to_unit_space(torch.cat((E0.unsqueeze(1),
                                               E1.unsqueeze(1),
                                               E2.unsqueeze(1),
                                               E3.unsqueeze(1),
                                               E4.unsqueeze(1),
                                               E5.unsqueeze(1),
                                               true_E), 1)).to(arg.device)
        y = torch.log10(true_E*10.).to(arg.device)
        x = logit_trafo(x)
        true_log_P = rec_model.log_prob(x, y).squeeze()
        print(true_log_P)
        for i in range(1000):
            x = trafo_to_unit_space(torch.cat((E0.unsqueeze(-1),
                                               E1.unsqueeze(-1),
                                               E2.unsqueeze(-1),
                                               E3.unsqueeze(-1),
                                               E4.unsqueeze(-1),
                                               E5.unsqueeze(-1),
                                               e_inc[i].unsqueeze(0)), 1)).to(arg.device)
            y = torch.log10(e_inc[i]*10.).to(arg.device)
            x = logit_trafo(x)
            log_P = rec_model.log_prob(x, y).squeeze()
            if torch.isnan(log_P): log_P = -500000*torch.ones(1).to(arg.device)
            else: log_P = log_P.unsqueeze(0)
            #print(i , log_P, log_P.shape, results.shape)
            results = torch.cat([results,log_P])
        if batch_id == 0:
            analysis_results = results
            #print(analysis_results.shape)
        else:
            analysis_results = torch.cat([analysis_results, results],-1)
            print(analysis_results.shape)
            
        if (batch_id+1) %10000 ==0: print('Analyzed {} %% of events'.format(int((batch_id+1)/10000)))

        if batch_id == 6: return analysis_results.detach().cpu().numpy()

################## train and evaluation functions for recursive flow ###############################

def train_rec_flow(rec_model, train_data, test_data, optim, arg):
    """ trains the flow that learns the energy distributions """
    best_eval_logprob_rec = float('-inf')

    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                       milestones=[5, 15, 40, 60, 120],
                                                       gamma=0.5,
                                                       verbose=True)

    num_epochs = 150
    for epoch in range(num_epochs):
        rec_model.train()
        for i, data in enumerate(train_data):

            E0 = data['layer_0_E']
            E1 = data['layer_1_E']
            E2 = data['layer_2_E']
            E3 = data['layer_3_E']
            E4 = data['layer_4_E']
            E5 = data['layer_5_E']
            E  = data['energy']
            y = torch.log10(E*10.).to(arg.device)
            if arg.log_preprocess:
                x = torch.cat((E0.unsqueeze(1),
                               E1.unsqueeze(1),
                               E2.unsqueeze(1),
                               E3.unsqueeze(1),
                               E4.unsqueeze(1),
                               E5.unsqueeze(1)), 1).to(arg.device)
                x = 2.*(torch.log10((x*1e5)+1e-8)-1.)
            else:
                x = trafo_to_unit_space(torch.cat((E0.unsqueeze(1),
                                                   E1.unsqueeze(1),
                                                   E2.unsqueeze(1),
                                                   E3.unsqueeze(1),
                                                   E4.unsqueeze(1),
                                                   E5.unsqueeze(1),
                                                   E), 1)).to(arg.device)
                x = logit_trafo(x)
            loss = - rec_model.log_prob(x, y).mean(0)
            #print(rec_model.log_prob(x, y))
            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % arg.log_interval == 0:
                print('Recursive Flow: epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, num_epochs, i, len(train_data), loss.item()))
                print('Recursive Flow: epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, num_epochs, i, len(train_data), loss.item()),
                      file=open(arg.results_file, 'a'))

        with torch.no_grad():
            rec_model.eval()
            loglike = []
            for data in test_data:
                E0 = data['layer_0_E']
                E1 = data['layer_1_E']
                E2 = data['layer_2_E']
                E3 = data['layer_3_E']
                E4 = data['layer_4_E']
                E5 = data['layer_5_E']
                E  = data['energy']
                y = torch.log10(E*10.).to(arg.device)
                if arg.log_preprocess:
                    x = torch.cat((E0.unsqueeze(1), E1.unsqueeze(1),
                                   E2.unsqueeze(1), E3.unsqueeze(1),
                                   E4.unsqueeze(1), E5.unsqueeze(1)), 1).to(arg.device)
                    x = 2.*(torch.log10((x*1e5)+1e-8)-1.)
                else:
                    x = trafo_to_unit_space(torch.cat((E0.unsqueeze(1),
                                                       E1.unsqueeze(1),
                                                       E2.unsqueeze(1),
                                                       E3.unsqueeze(1),
                                                       E4.unsqueeze(1),
                                                       E5.unsqueeze(1),
                                                       E), 1)).to(arg.device)

                    x = logit_trafo(x)

                loglike.append(rec_model.log_prob(x, y))

            logprobs = torch.cat(loglike, dim=0).to(arg.device)

            logprob_mean = logprobs.mean(0)
            logprob_std = logprobs.var(0).sqrt()# / np.sqrt(len(test_data.dataset))

            output = 'Recursive Flow: Evaluate (epoch {}) -- '.format(epoch+1) +\
                'logp(x, at E(x)) = {:.3f} +/- {:.3f}'

            print(output.format(logprob_mean, logprob_std))
            print(output.format(logprob_mean, logprob_std), file=open(arg.results_file, 'a'))
            eval_logprob_rec = logprob_mean
        lr_schedule.step()
        if eval_logprob_rec > best_eval_logprob_rec:
            best_eval_logprob_rec = eval_logprob_rec
            save_rec_flow(rec_model, arg)

@torch.no_grad()
def sample_rec_flow(rec_model, num_pts, arg, energies):
    """ samples layer energies for given total energy from rec flow """
    rec_model.eval()
    context = torch.log10(energies*10.).to(arg.device)
    samples = rec_model.sample(num_pts, context.unsqueeze(-1))
    if arg.log_preprocess:
        return samples.squeeze()
    else:
        return inverse_logit(samples.squeeze())


####################################################################################################
#######################################   running the code   #######################################
####################################################################################################

if __name__ == '__main__':
    args = parser.parse_args()

    # check if parsed arguments are valid
    assert (args.train or args.generate or args.evaluate or args.generate_to_file or \
            args.save_only_weights or args.evaluate_KL or args.analyze), \
            ("Please specify at least one of --train, --generate, --evaluate, --generate_to_file")

    # check if output_dir exists and 'move' results file there
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    args.results_file = os.path.join(args.output_dir, args.results_file)
    print(args, file=open(args.results_file, 'a'))

    if args.fully_guided:
        args.train_p = True
        args.train_xz = True
    # setup device
    args.device = torch.device('cuda:'+str(args.which_cuda) \
                               if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print("Using {}".format(args.device))
    print("Using {}".format(args.device), file=open(args.results_file, 'a'))


    # get dataloaders needed for training / evaluation
    if (args.train or args.evaluate or args.evaluate_KL):
        if args.data_dir is None:
            raise ValueError("--data_dir needs to be specified!")
        train_dataloader, test_dataloader = get_dataloader(args.particle_type,
                                                           args.data_dir,
                                                           full=False,
                                                           apply_logit=True,
                                                           device=args.device,
                                                           batch_size=args.batch_size,
                                                           with_noise=args.with_noise,
                                                           normed=False,
                                                           normed_layer=True)

    args.input_size = {'0': 288, '1': 144, '2': 72, '3': 288, '4': 144, '5': 72}
    args.input_dims = {'0': (3, 96), '1': (12, 12), '2': (12, 6), '3': (3, 96), '4': (12, 12), '5': (12, 6)}

    flow_params_rec_energy = {'num_blocks': 2, #num of layers per block
                              'features': 6,
                              'context_features': 1,
                              'hidden_features': 64,
                              'use_residual_blocks': False,
                              'use_batch_norm': False,
                              'dropout_probability': 0.,
                              'activation':getattr(F, args.activation_fn),
                              'random_mask': False,
                              'num_bins': 8,
                              'tails':'linear',
                              'tail_bound': 14,
                              'min_bin_width': 1e-6,
                              'min_bin_height': 1e-6,
                              'min_derivative': 1e-6}
    rec_flow_blocks = []
    for _ in range(6):
        rec_flow_blocks.append(
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                **flow_params_rec_energy))
        rec_flow_blocks.append(transforms.RandomPermutation(6))
    rec_flow_transform = transforms.CompositeTransform(rec_flow_blocks)
    rec_flow_base_distribution = distributions.StandardNormal(shape=[6])
    rec_flow = flows.Flow(transform=rec_flow_transform, distribution=rec_flow_base_distribution)

    rec_model = rec_flow.to(args.device)
    rec_optimizer = torch.optim.Adam(rec_model.parameters(), lr=1e-4)
    print(rec_model)
    print(rec_model, file=open(args.results_file, 'a'))

    total_parameters = sum(p.numel() for p in rec_model.parameters() if p.requires_grad)

    print("Recursive energy setup has {} parameters".format(int(total_parameters)))
    print("Recursive energy setup has {} parameters".format(int(total_parameters)),
          file=open(args.results_file, 'a'))

    # check if Flow I checkpoint exists, either load it or train new one.
    #if os.path.exists(os.path.join(args.output_dir, args.particle_type+'_rec.pt')):
    if args.flowI_restore_file is None:
        flowI_file = os.path.join(args.output_dir,
                                  f"saved_checkpoints/Flow_I/{args.particle_type}_sep19.pt")
    else:
        flowI_file = args.flowI_restore_file
    if os.path.exists(flowI_file):
        print("loading recursive energy flow")
        print("loading recursive energy flow", file=open(args.results_file, 'a'))
        load_rec_flow(rec_model, args)
    else:
        train_rec_flow(rec_model, train_dataloader, test_dataloader, rec_optimizer, args)
        print("loading recursive energy flow")
        print("loading recursive energy flow", file=open(args.results_file, 'a'))
        load_rec_flow(rec_model, args)

    ## test rec_flow:
    #num_pts = 50000
    #energies = 0.99*torch.rand((num_pts,)) + 0.01
    #testsamples = sample_rec_flow(rec_model, 1, args, energies).to('cpu')
    #np.save(os.path.join(args.output_dir, 'rec_flow_samples.npy'), testsamples.numpy())
    #testsamples = trafo_to_energy_space(testsamples, energies)*1e5
    #testsamples_large = torch.zeros((num_pts, 504))
    #testsamples_large[:, 0] = testsamples[:, 0]
    #testsamples_large[:, 288] = testsamples[:, 1]
    #testsamples_large[:, 432] = testsamples[:, 2]
    #args.dim_split = [288, 144, 72]
    #plot_all(testsamples_large, args, used_energies=energies.reshape(-1, 1))

    # to plot losses:
    args.train_loss = []
    args.test_loss = []
    # to keep track of dimensionality in constructing the flows
    args.dim_sum = 0
    args.dim_split = []

    flow_params_RQS = {'num_blocks':args.n_hidden, # num of hidden layers per block
                       'use_residual_blocks':False,
                       'use_batch_norm':False,
                       'dropout_probability':args.dropout_probability,
                       'activation':getattr(F, args.activation_fn),
                       'random_mask':False,
                       'num_bins':args.n_bins,
                       'tails':'linear',
                       'tail_bound':args.tail_bound,
                       'min_bin_width': 1e-6,
                       'min_bin_height': 1e-6,
                       'min_derivative': 1e-6}

    # setup flow
    flow_blocks = []
    for layer_id in range(6):
        current_dim = args.input_size[str(layer_id)]
        args.dim_split.append(current_dim)
    for entry in args.dim_split:
        args.dim_sum += entry
    cond_label_size = 7
    for i in range(args.n_blocks):
        flow_blocks.append(
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                **flow_params_RQS,
                features=args.dim_sum,
                context_features=cond_label_size,
                hidden_features=args.hidden_size
                ))

        if i%2 == 0:
            flow_blocks.append(InversionLayer(args.dim_split))
        else:
            flow_blocks.append(RandomPermutationLayer(args.dim_split))

    del flow_blocks[-1]
    if args.student_mode:
        flow_transform = GuidedCompositeTransform(flow_blocks)
    else:
        flow_transform = transforms.CompositeTransform(flow_blocks)

    flow_base_distribution = distributions.StandardNormal(shape=[args.dim_sum])
    flow = flows.Flow(transform=flow_transform, distribution=flow_base_distribution)

    model = flow.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(model)
    print(model, file=open(args.results_file, 'a'))

    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total setup has {} parameters".format(int(total_parameters)))
    print("Total setup has {} parameters".format(int(total_parameters)),
          file=open(args.results_file, 'a'))


    if not args.student_mode:
        # run in teacher mode
        print("Running in teacher mode.")
        if args.train:
            print("training teacher ...")
            train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, args,
                               rec_model=rec_model)

        if args.evaluate:
            print("evaluating teacher ...")
            load_weights(model, args)
            evaluate(model, test_dataloader, args.n_epochs, args)

        if args.generate_to_file and not args.generate:
            print("generating from teacher to file ...")
            load_weights(model, args)
            # for nn plots
            #my_energies = torch.tensor(2000*[0.05, 0.1, 0.2, 0.5, 0.95])
            generate_to_file(model, args, rec_model, num_events=100000, energies=None) #my_energies

        if args.analyze:
            print("performing MLE analysis ...")
            load_weights(model, args)
            analysis_results = MLE_analysis(model, args, rec_model)
            np.savetxt('analysis_results.txt', analysis_results)

        if args.save_only_weights:
            print("saving only teacher weights ...")
            load_all(model, optimizer, args)
            save_weights(model, args)
    else:
        # run in student mode
        print("Running in student mode.")
        if args.train or args.evaluate_KL:
            print("loading teacher")
            teacher = model
            load_weights(teacher, args)
            print("done")
        del model

        # to plot losses:
        args.train_loss = []
        args.test_loss = []

        # setup student for training
        if args.train_xz or args.train_p:
            teacher_perm = []
        else:
            teacher_perm = torch.arange(0, args.dim_sum)
        if args.train:
            # properly treat teacher permutations when training from scratch
            for elem in teacher._transform._transforms:
                if hasattr(elem, '_permutation'):
                    if args.train_xz or args.train_p:
                        teacher_perm.append(elem._permutation.to('cpu'))
                    else:
                        teacher_perm = torch.index_select(teacher_perm, 0,
                                                          elem._permutation.to('cpu'))
        else:
            # fill with dummies that are then overwritten in loading.
            if args.train_xz or args.train_p:
                for _ in range(args.n_blocks-1 if (args.train_xz or args.train_p) \
                               else args.student_n_blocks-1):
                    teacher_perm.append(transforms.Permutation(torch.arange(0, args.dim_sum)))
            else:
                teacher_perm = transforms.Permutation(torch.arange(0, args.dim_sum))
        if args.train_xz or args.train_p:
            teacher_perm.append(teacher_perm[-1])
        flow_blocks = []
        student_perms = []
        for i in range(args.n_blocks if (args.train_xz or args.train_p) \
                       else args.student_n_blocks):
            flow_blocks.append(
                transforms.InverseTransform(
                    #transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    IAFRQS(
                        **flow_params_RQS,
                        features=args.dim_sum,
                        context_features=cond_label_size,
                        hidden_features=args.student_hidden_size
                    )))

            if i%2 == 0:
                flow_blocks.append(InversionLayer(args.dim_split))
            else:
                if args.train:
                    if args.train_xz or args.train_p:
                        flow_blocks.append(transforms.Permutation(teacher_perm[i]))
                    else:
                        flow_blocks.append(RandomPermutationLayer(args.dim_split))
                else:
                    # add dummy permutation that will be overwritten with loaded model
                    flow_blocks.append(transforms.Permutation(torch.arange(0, args.dim_sum)))
            student_perms.append(flow_blocks[-1]._permutation)
        del flow_blocks[-1]
        del student_perms[-1]
        if not (args.train_xz or args.train_p):
            if not args.train:
                # overwrite teacher_perm, so teacher model is not needed, unless in training
                teacher_perm = torch.arange(0, args.dim_sum)
            student_perms.reverse()
            final_perm = torch.arange(0, args.dim_sum)
            for perm in student_perms:
                final_perm = torch.index_select(final_perm, 0, torch.argsort(perm))
            final_perm = torch.index_select(final_perm, 0, teacher_perm)
            flow_blocks.append(transforms.Permutation(final_perm))
            flow_transform = transforms.CompositeTransform(flow_blocks)
        else:
            flow_transform = GuidedCompositeTransform(flow_blocks)

        flow_base_distribution = distributions.StandardNormal(shape=[args.dim_sum])
        student = flows.Flow(transform=flow_transform,
                             distribution=flow_base_distribution).to(args.device)

        optimizer_student = torch.optim.Adam(student.parameters(), lr=args.lr)
        print(student)
        print(student, file=open(args.results_file, 'a'))

        total_parameters = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print("Student has {} parameters".format(int(total_parameters)))
        print("Student has {} parameters".format(int(total_parameters)),
              file=open(args.results_file, 'a'))

        if args.train:
            print("training student ...")
            train_and_evaluate_student(teacher, student, train_dataloader, test_dataloader,
                                       optimizer_student, args, rec_model=rec_model)

        if args.generate:
            print("generating from student ...")
            load_weights(student, args, is_student=True)
            generate(student, args, step=args.n_epochs, include_average=True, rec_model=rec_model)

        if args.evaluate_KL:
            print("evaluating student KL ...")
            load_weights(student, args, is_student=True)
            evaluate_KL(student, teacher, test_dataloader, args)
        if args.evaluate:
            print("evaluating student ...")
            load_weights(student, args, is_student=True)
            evaluate(student, test_dataloader, args.n_epochs, args)

        if args.generate_to_file and not args.generate:
            print("generating from student to file ...")
            load_weights(student, args, is_student=True)
            # for nn plots
            #my_energies = torch.tensor(2000*[0.05, 0.1, 0.2, 0.5, 0.95])
            generate_to_file(student, args, num_events=100000, energies=None, #my_energies, #None
                             rec_model=rec_model)

        if args.save_only_weights:
            print("saving only student weights ...")
            load_all(student, optimizer, args, is_student=True)
            save_weights(student, args, is_student=True)



# This code was written under the influence of https://www.youtube.com/watch?v=0KGATvoIyRc \m/
