# Design_Robust_Network
The repository contains the codes and data for the manuscript "Design of robust networks via reinforcement learning prompt the emergence of multi-backbones".

# Overview
We have provided the code for training, network design, and backbone analysis, along with the training data and corresponding design results. Due to the large file sizes,the initial data, the design results and trained models have been compressed in .7z format and are available in the release section.

# Initial Network Data
The synthetic and real network datasets used for the design tasks are available in the `Data.7z` file in the Release section. They should be downloaded and extracted into the `./Data` directory.

# Initialize Env
1. Install the required packages based on the **requiements** file.
2. If one would like to train a design policy for MS or GND(R) attacks, the attack executable file needs to be recomplied in 
"\dismantlers\decycler-master" or "dismantlers\Generalized-Network-Dismantling-Input" following the instructions from "https://github.com/abraunst/decycler" or "https://github.com/renxiaolong/Generalized-Network-Dismantling".
3. If one would like to train a design policy for GDM attacks, the env and attack executable file should be set following the instructions from "[https://github.com/abraunst/decycler](https://github.com/NetworkScienceLab/GDM)"
4. Activate the Env.

# Train
```bash
cd RL_Algorithm
python train.py
python train_cost.py
```

# Design
```bash
cd RL_Algorithm
python Design_synthetic_networks.py
python Design_large_networks.py
python Design_real_networks.py
python Design_real_networks_cost
```

# Reproduce main results
## Data and Model Preparation
- The synthetic and real network datasets are available in the `Data.7z` file (Release section).  
  Please extract it into the `./Data` directory.
- The trained models and designed network results are provided in `Trained_models.7z` and `Design_result.7z` (Release section).  
  Before reproducing results, please extract these two files into their corresponding subfolders `/Trained_models` and `/Design_result` under the `/RL_Algorithm` directory.

## Design cases on synthetic and real-world networks
```bash
cd RL_Algorithm
python Case_of_synthetics.py
python Case_of_Real_Networks.py
```

## Silhouette_Analysis
```bash
cd Silhouette_Analysis
python backbone_for_100.py
python backbone_for_600.py
python backbone_for_Real.py
python backbone_Statistic.py
```

# Issues
If you find a problem with the implementation code, please ask under issue or contact us.
