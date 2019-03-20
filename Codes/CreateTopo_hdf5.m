clear;
files = dir('Topographies/*Baseline*.mat');
filenames = extractfield(files,'name');
filenames = strcat('Topographies/',filenames);
tic
topo_tensor = zeros(size(files,1),2500,3,64,64);
for f=1:size(files,1)
    temp_topo = load(char(filenames(f)));
    topo_tensor(f,:,:,:,:) = temp_topo.topographies;
end
hdf5write('Baseline_allTopo.h5','/home',topo_tensor);
toc