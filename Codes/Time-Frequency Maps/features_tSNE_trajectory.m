%close all;
data_path = '';%'/home/arna/Project/EEG/TF_maps/';
features = hdf5read([data_path,'Layer3_baseline.h5'],'/home/features')';
Labels = hdf5read([data_path,'Layer3_baseline.h5'],'/home/Labels')';
SubjLabels = hdf5read([data_path,'Layer3_baseline.h5'],'/home/SubjLabels')';
features_downsampled = squeeze(nanmean(reshape(features,7500,size(features,1)/7500,[])));
Labels = downsample(Labels,7500);
SubjLabels = downsample(SubjLabels,7500);
features_downsampled0 = features_downsampled;
Labels0 = Labels;
SubjLabels0 = SubjLabels;

features = hdf5read([data_path,'Layer3_30min.h5'],'/home/features')';
Labels = hdf5read([data_path,'Layer3_30min.h5'],'/home/Labels')';
SubjLabels = hdf5read([data_path,'Layer3_30min.h5'],'/home/SubjLabels')';
features_downsampled = squeeze(nanmean(reshape(features,7500,size(features,1)/7500,[])));
Labels = downsample(Labels,7500);
SubjLabels = downsample(SubjLabels,7500);
features_downsampled1 = features_downsampled;
Labels1 = 2+Labels;
SubjLabels1 = 25+SubjLabels;

features = hdf5read([data_path,'Layer3_60min.h5'],'/home/features')';
Labels = hdf5read([data_path,'Layer3_60min.h5'],'/home/Labels')';
SubjLabels = hdf5read([data_path,'Layer3_60min.h5'],'/home/SubjLabels')';
features_downsampled = squeeze(nanmean(reshape(features,7500,size(features,1)/7500,[])));
Labels = downsample(Labels,7500);
SubjLabels = downsample(SubjLabels,7500);
features_downsampled2 = features_downsampled;
Labels2 = 4+Labels;
SubjLabels2 = 50+SubjLabels;

features = hdf5read([data_path,'Layer3_90min.h5'],'/home/features')';
Labels = hdf5read([data_path,'Layer3_90min.h5'],'/home/Labels')';
SubjLabels = hdf5read([data_path,'Layer3_90min.h5'],'/home/SubjLabels')';
features_downsampled = squeeze(nanmean(reshape(features,7500,size(features,1)/7500,[])));
Labels = downsample(Labels,7500);
SubjLabels = downsample(SubjLabels,7500);
features_downsampled3 = features_downsampled;
Labels3 = 6+Labels;
SubjLabels3 = 75+SubjLabels;

%{
features = hdf5read('/home/arna/Project/EEG/Topo_Maps/Layer3.h5','/home/features')';
Labels = hdf5read('/home/arna/Project/EEG/Topo_Maps/Layer3.h5','/home/Labels')';
SubjLabels = hdf5read('/home/arna/Project/EEG/Topo_Maps/Layer3.h5','/home/SubjLabels')';
%}

%features = downsample(features,25); % downsample by rejection
% downsample by meaning
%features_downsampled(:,1) = nanmean(reshape([features(:,1); nan(mod(-numel(features(:,1)),25),1)],25,[]))';
%features_downsampled(:,2) = nanmean(reshape([features(:,2); nan(mod(-numel(features(:,2)),25),1)],25,[]))';

%figure;
%subplot(2,1,1);gscatter(features_downsampled(:,1),features_downsampled(:,2),Labels); title('Last Layer Outputs with Group Labels');
%subplot(2,1,2);gscatter(features_downsampled(:,1),features_downsampled(:,2),SubjLabels); title('Last Layer Outputs with Subject Labels');
features_downsampled = [features_downsampled0;features_downsampled1;features_downsampled2;features_downsampled3];
Labels = [Labels0,Labels1,Labels2,Labels3]';
SubjLabels = [SubjLabels0,SubjLabels1,SubjLabels2,SubjLabels3];
Labels_text={'Baseline CON';'Baseline EXE';'30min CON';'30min EXE';'60min CON';'60min EXE';'90min CON';'90min EXE'};
%Labels_text={'CON';'EXE';'CON';'EXE';'CON';'EXE';'CON';'EXE'};
Labels = Labels_text(Labels);
tic
rng default
tSNE_features = tsne(features_downsampled);
toc
figure;
subplot(1,1,1);hh1 = gscatter(tSNE_features(:,1),tSNE_features(:,2),Labels,'rrggbbmm','xo'); 
%subplot(1,1,1);hh1 = gscatter(tSNE_features(:,1),tSNE_features(:,2),Labels,'ck','oo^^ddxx'); 
title('tSNE plots for Layer 3 Outputs with Group Labels');
%subplot(2,1,2);hh2 = gscatter(tSNE_features(:,1),tSNE_features(:,2),SubjLabels); 
%title('tSNE plots for Layer 3 Outputs with Subject Labels');