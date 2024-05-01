# Unifying Simulation and Inference with Normalizing Flows
## by Haoxing Du, Claudius Krause, Vinicius Mikuni, Benjamin Nachman, Ian Pang and David Shih 
This repository contains the source code used to produce the results of

_"Unifying Simulation and Inference with Normalizing Flows"_ by Haoxing Du, Claudius Krause, Vinicius Mikuni, Benjamin Nachman, Ian Pang and David Shih, [arxiv: 2404.18992]

### Detector Layout and Training Data
We consider a new sampling calorimeter (ECAL+HCAL) version of the toy detector used in the original [CaloGAN](https://arxiv.org/abs/1712.10321). This new calorimeter setup includes a HCAL which was not included in the setup used the [most recent CaloGAN update](https://zenodo.org/records/10393540). The original dataset included energy contributions from both active and inactive calorimeter layers, whereas our new dataset only includes energy contributions from the active layers as would be available in practice. In our calorimeter setup, the sampling fractions for the ECAL and HCAL are $\sim20$% and $\sim 1.3$% respectively. Like the original toy detector, we have a three-layer ECAL. However, we also include a three-layer HCAL positioned behind the ECAL. The six layers have a voxel resolution of $`3\times 96`$, $`12\times 12`$, $`12\times 6`$, $`3\times 96`$, $`12\times 12`$, and $`12\times 6`$, respectively. 

The new dataset can be found at [https://zenodo.org/records/11073232](https://zenodo.org/records/11073232).

### Training CaloFlow
Please see https://gitlab.com/claudius-krause/caloflow for instructions on training CaloFlow.

### Computing likelihood for calibration task
To use trained flows to compute likelihood for calibration task, run

`python MLE_analysis-100k_densities.py --weights_dir =FLOW_WEIGHTS_DIR_NAME --results_dir=RESULTS_DIR_NAME`
