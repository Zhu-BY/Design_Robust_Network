# Design_Robust_Network
The repository contains the codes and data for the manuscript "Design of robust networks via reinforcement learning prompt the emergence of multi-backbones".

# Overview
We have provided the code for training, network design, and backbone analysis, along with the training data and corresponding design results. Due to the large file sizes, the design results and trained models have been compressed in 7z format and are available in the release section.

# Initial Network Data
The synthetic and real network datasets used for the design tasks are located in the `Data` folder.  
To access the data, please navigate to the `Data` directory and execute the following commands:

```bash
cd Data
cat data.tar.gz.* | tar xzvf -


# Code
For implementation, please follow steps below:
1. Install the required packages based on the **requiements** file.
2. Confirme the file path and training parameters including the attack strategies.
3. If you would like to train a design policy for MS or GND(R) attacks, the attack executable file needs to be recomplied in 
"\dismantlers\decycler-master" or "dismantlers\Generalized-Network-Dismantling-Input" following the instructions from "https://github.com/abraunst/decycler" or "https://github.com/renxiaolong/Generalized-Network-Dismantling".
4. run main,py

# Issues
If you find a problem with the implementation code, please ask under issue or contact us.
