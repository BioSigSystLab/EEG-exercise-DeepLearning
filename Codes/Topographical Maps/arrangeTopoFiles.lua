require 'hdf5';
File = hdf5.open('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/Baseline_allTopo.h5','r');
data = File:read('/home'):all()

for i=1,torch.floor(data:size():size()/2) do
	data = data:transpose(i,data:size():size()-i+1);
end

data = data:contiguous();
torch.save('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/Baseline_allTopo.t7',data)
print('baseline done')

File = hdf5.open('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/90min_allTopo.h5','r');
data = File:read('/home'):all()

for i=1,torch.floor(data:size():size()/2) do
	data = data:transpose(i,data:size():size()-i+1);
end

data = data:contiguous();
torch.save('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/90min_allTopo.t7',data)
print('90min done')