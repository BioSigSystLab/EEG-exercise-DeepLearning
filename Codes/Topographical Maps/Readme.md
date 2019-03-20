# The codes used for analysis of Topographical maps

The file descriptions are listed in the order they were run (or should be run for future analysis on new data) :-
* **arrangeTopoFiles.lua** - Reorganizes the hdf5 files containing topographies because they were generated from MATLAB. Reversing the order of the axes makes it suitable to be loaded in Torch.
* **lsm_model.lua** - Contains description of the BaseCNN. *no need to run this file as it is called within train_XX files*
* **lsm_topmodel.lua** - Contains description of the TopNN/TopCNN. *no need to run this file as it is called within train_XX files*
* **train_vol_lsm.lua** - Trains the model on a particular portion of topo data, does not do subject cross-validation. Returns a model that is trained on 80% of the entire data (early-stopping used to avoid over-fitting). 20% data used as validation and testing set.
* **train_vol_lsm_subj.lua** - Evaluates 10-fold CV with 23 subjects in training set and 2 subjects in validation/testing set.
* **ccGAP.lua** - Loads a pre-trained model and runs the ccGAP algorithm to estimate filter importances. These importances are later used to generate activation maps.
* **ccGAP_deploy.lua** - Uses the estimated filter importances to generate activation maps for the CON and EXE classes from the entire dataset.
* **visualize_ccGAP.m** - Visualizes the generated activation maps and do statistics between groups.
* **featureVec_generate.lua** - Saves a hdf5 file containing the feature maps (or layer representations) of each input sample in the dataset, that is used by *motor_learning_scores_corr.m* and *features_tSNE_trajectory.m* files.
