# Design_Robust_Network
The repository contains the codes for the manuscript "Design of robust networks via reinforcement learning prompt the emergence of multi-backbones".

# Overview
For code, we provide the codes of our RL framework and the attack pool.
For data, we provide two real networks experimented in our work.
# Data
The topology data of Germany grid is from "https://icon.colorado.edu/networks", and the data of Sprintlink ISP network is from "https://research.cs.washington.edu/networking/rocketfuel/interactive/".
# Code
For implementation, please follow steps below:
1. Install the required packages based on the **requiements** file.
2. Confirme the file path and training parameters including the attack strategies.
3. If you would like to train a design policy for MS or GND(R) attacks, the attack executable file needs to be recomplied in 
"\dismantlers\decycler-master" or "dismantlers\Generalized-Network-Dismantling-Input" following the instructions from "https://github.com/abraunst/decycler" or "https://github.com/renxiaolong/Generalized-Network-Dismantling".
4. run main,py

# Issues
If you find a problem with the implementation code, please ask under issue or contact us.
