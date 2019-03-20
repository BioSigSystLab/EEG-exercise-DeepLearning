%close all;

%{
features = hdf5read('/home/arna/Project/EEG/TF_maps/Layer3_30min.h5','/home/features')';
Labels = hdf5read('/home/arna/Project/EEG/TF_maps/Layer3_30min.h5','/home/Labels')';
SubjLabels = hdf5read('/home/arna/Project/EEG/TF_maps/Layer3_30min.h5','/home/SubjLabels')';
%}

features = hdf5read('/home/arna/Project/EEG/Topo_Maps/Layer3.h5','/home/features')';
Labels = hdf5read('/home/arna/Project/EEG/Topo_Maps/Layer3.h5','/home/Labels')';
SubjLabels = hdf5read('/home/arna/Project/EEG/Topo_Maps/Layer3.h5','/home/SubjLabels')';

%features = downsample(features,25); % downsample by rejection
% downsample by meaning
%features_downsampled(:,1) = nanmean(reshape([features(:,1); nan(mod(-numel(features(:,1)),25),1)],25,[]))';
%features_downsampled(:,2) = nanmean(reshape([features(:,2); nan(mod(-numel(features(:,2)),25),1)],25,[]))';
features_downsampled = squeeze(nanmean(reshape(features,25,size(features,1)/25,[])));
Labels = downsample(Labels,25);
SubjLabels = downsample(SubjLabels,25);

%figure;
%subplot(2,1,1);gscatter(features_downsampled(:,1),features_downsampled(:,2),Labels); title('Last Layer Outputs with Group Labels');
%subplot(2,1,2);gscatter(features_downsampled(:,1),features_downsampled(:,2),SubjLabels); title('Last Layer Outputs with Subject Labels');

tic
rng default
tSNE_features = tsne(features_downsampled);
toc
figure;
subplot(2,1,1);hh1 = gscatter(tSNE_features(:,1),tSNE_features(:,2),Labels); 
title('tSNE plots for Layer 3 Outputs with Group Labels');
subplot(2,1,2);hh2 = gscatter(tSNE_features(:,1),tSNE_features(:,2),SubjLabels); 
title('tSNE plots for Layer 3 Outputs with Subject Labels');