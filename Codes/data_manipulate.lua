require 'nn';
require 'image';
require 'optim';
require 'gnuplot';
local matio = require 'matio';

--model = require 'model.lua'

train = matio.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/EEG_dat.mat')

Data = train.A:contiguous()
print(Data:size())

baseline = torch.Tensor(25,64,12501,55)
min_30 = torch.Tensor(25,64,12501,55)
min_60 = torch.Tensor(25,64,12501,55)
min_90 = torch.Tensor(25,64,12501,55)

print("reading done")
for i=1,25 do
	baseline[i] = Data[4*i]
	min_30[i] = Data[4*i-3]
	min_60[i] = Data[4*i-2]
	min_90[i] = Data[4*i-1]
end

print("saving now")
torch.save('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/torch_baseline.t7',baseline)
torch.save('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/torch_min30.t7',min_30)
torch.save('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/torch_min60.t7',min_60)
torch.save('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/torch_min90.t7',min_90)