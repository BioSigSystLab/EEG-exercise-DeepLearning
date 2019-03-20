# Description of codes

## Requirements
1. Torch7
2. MATLAB
3. [Shaded Error bar MATLAB toolbox](https://github.com/raacampbell/shadedErrorBar)
4. Preferably a *GPU* to aid computation speed -- *If not, edit the codes to remove the word ``cuda''*

## A brief outline
* **data_manipulate.lua** - splits the entire data matrix into baseline, 30min, 60min and 90min sessions.
* **CreateTF_hdf5.m** - Creates separate hdf5 files for time-frequency data of each session.
* **CreateTopographies.m** - Project the ERD information from desired frequency band to topographical space.
* **CreateTopo_hdf5.m** - Creates separate hdf5 files for topographical data of each session.
* **Time-Frequency Maps** - Contains files required for analysis of Time-frequency data.
* **Topographical Maps** - Contains files required for analysis of Topographical data.
