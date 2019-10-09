# The codes used for analysis of Time-frequency maps

The file descriptions are listed in the order they were run (or should be run for future analysis on new data) :-
* **lsm_model.lua** - Contains description of the BaseCNN. *no need to run this file as it is called within train_XX files*
* **lsm_topmodel.lua** - Contains description of the TopNN/TopCNN. *no need to run this file as it is called within train_XX files*
* **train_lsm.lua** - Trains the model on a particular portion of data, does not do subject cross-validation. Returns a model that is trained on 80% of the entire data (early-stopping used to avoid over-fitting). 20% data used as validation and testing set.
* **train_lsm_subj.lua** - Evaluates 10-fold CV with 23 subjects in training set and 2 subjects in validation/testing set.
* **ccGAP.lua** - Loads a pre-trained model and runs the ccGAP algorithm to estimate filter importances. These importances are later used to generate activation maps.
* **ccGAP_deploy.lua** - Uses the estimated filter importances to generate activation maps for the CON and EXE classes from the entire dataset.
* **featureVec_generate.lua** - Saves a hdf5 file containing the feature maps (or layer representations) of each input sample in the dataset, that is used by *motor_learning_scores_corr.m* and *features_tSNE_trajectory.m* files.
* **motor_learning_scores_corr.m** - Requires the feature maps from a trained model to be saved in a hdf5 file before running this *(See Line 16 of this file)*. Calculates the correlation coefficient of the features with motor learning scores.
* **features_tSNEplots.m** - Plots the layer representations in a lower dimensional to show the group and subject discriminability.
* **features_tSNE_trajectory.m** - Plots the trajectory of layer representations (or feature maps) through time in a lower 2D dimension.
---------------
* **python_RF.py, CSP_RF.py** - Implements RF and CSP to get baseline EEG decoding results
* **GAP.lua, GAP_deploy.lua, GradCAM.lua** - Implements GAP and gradCAM to compare their performance against the ccGAP results
