REMEMBER to work in a different branch than main and then send a pull request in order to merge it into the final code.

TO CREATE CONDA ENV WITH ALL DEPENDENCIES:
create a folder in your repository called "env"
use 'conda env create -f environment.yml -p INSERT_PATH_TO_REPO/env' (on Unix-based systems)
or 'conda env create -f environment.yml -p INSERT_PATH_TO_REPO\env' (on Windows)

Possible Values for laser:
Laser freq ~ 800nm; 1.57 eV = w_L = 0.057 au
I_p ~ 10eV (13.6 eV)
maybe I = 10^14 W/cm^2
