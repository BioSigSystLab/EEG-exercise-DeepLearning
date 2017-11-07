# Description of codes
+ AccVsTimeCurves.m - Plot classification confidence of network for each timepoint averaged over all subjects of a group.
+ analyseLinearWeights.m - Visualise the linear weights learnt by the network
+ analyseTopNN.m - Visualise the Top NN weights.
+ avgGroup_filter_vis.m - Visualise the Class Activation Map (using AdGAP) of the input from different regions of brain averaged over all subjects of group.
+ filter_vis.m - Visualise the convolution operation learnt by the network. Plots the different filter response and the band of interest of each filter.
+ basicTrain.lua - A simple network to train avg morlet wavelet components of entire session of EEG data into exercise and control.
+ data_manipulate.lua - Arranges data from each subject into different files corresponding to baseline, 30min, 60min and 90min to load less data while training.
+ fineTune_lsm.lua - Takes baseCNN trained on 90min data(say) and finetunes topCNN for some other session data. Achieves marginally better accuracy than training from scratch.
+ lsm_deploy.lua - Deploy lsm to test accuracy for entire dataset.
+ lsm_deploy_timeAcc.lua - Deploys lsm to return a plot of confidence for each timepoint in trial for each subject.
+ lsm_model.lua - baseCNN model architecture
+ lsm_topmodel.lua - topNN model architecture
+ model.lua - simple NN architecture for direct classification
+ modified_GAP.lua - AdGAP code for filter weights and class activation map
+ train_lsm.lua - train the LSM model ot classify into control and exercise.
