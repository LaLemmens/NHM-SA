# NHM-SA
Multiphase simulated annealing reconstruction algorithm.

This folder contains the script and training image used to generate the cement past realizations found in Lemmens, L., Rogiers, B., Jacques, D., Huysmans, M., Swennen, R., Urai, J. L., ... & Laloy, E. (2019). Nested multiresolution hierarchical simulated annealing algorithm for porous media reconstruction. Physical Review E, 100(5), 053316.

The file NMHSA contains all functions used to run code the function NMH_SA represents the 2D version of the code and the function NMH_SA2Dto3D the 3D version. The name of the input variables is described in the function itself.
The wrapper calls the functions from NMHSA.py with the respective input to generate the realizations represented in the paper of our code with the variable setting that was used.


