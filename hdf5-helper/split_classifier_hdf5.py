# pylint: disable=invalid-name
""" Takes one .hdf5 file as input and splits it in 3 smaller files for training/test/validation of
    CaloFlow and the classifier metric.

    This is step 1 in creating the dataset for the classifier test:
    1. split generated file in train/test/val set (this file)
    2. merge the generated sets with the corresponding GEANT4 sets (merge_hdf5.py)

    The split of 100k events is done as follows:
        - 60k for training the classifier
        - 20k for validating the classifier (model selection and iso_reg calibration)
        - 20k for getting the numerical results of the model quality

    This code was used for the following publications:

    "CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows"
    by Claudius Krause and David Shih
    arxiv:2106.05285, Phys.Rev.D 107 (2023) 11, 113003

    "CaloFlow II: Even Faster and Still Accurate Generation of Calorimeter Showers with
     Normalizing Flows"
    by Claudius Krause and David Shih
    arXiv:2110.11377, Phys.Rev.D 107 (2023) 11, 113004

"""

import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file', '-f', help='File to be split')

def look_at_energies(data_array):
    print("Energy has length {}".format(len(data_array)))
    print("Mean is {} +/- {}".format(data_array.mean(), data_array.std()))
    print('* * *')

args = parser.parse_args()

input_file = h5py.File(args.file, 'r')

key_len = []
for key in input_file.keys():
    key_len.append(len(input_file[key]))
key_len = np.array(key_len)

assert np.all(key_len == key_len[0])
assert key_len[0] == 100000

cut_idx = [0, 60000, 80000, 100000]

train_cls_file = h5py.File('train_cls_'+args.file, 'w')
test_cls_file = h5py.File('test_cls_'+args.file, 'w')
val_cls_file = h5py.File('val_cls_'+args.file, 'w')

for key in input_file.keys():
    train_cls_file.create_dataset(key, data=input_file[key][cut_idx[0]:cut_idx[1]])
    test_cls_file.create_dataset(key, data=input_file[key][cut_idx[1]:cut_idx[2]])
    val_cls_file.create_dataset(key, data=input_file[key][cut_idx[2]:cut_idx[3]])

for dataset in [input_file, train_cls_file, test_cls_file, val_cls_file]:
    look_at_energies(dataset['energy'][:])

train_cls_file.close()
test_cls_file.close()
val_cls_file.close()
