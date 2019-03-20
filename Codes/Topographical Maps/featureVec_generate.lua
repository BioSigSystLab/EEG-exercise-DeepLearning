require 'nn';
require 'image';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';
require 'hdf5';

model = torch.load('Saved Models/lsm_model_90min_lambda6.t7');
--model = torch.load('Saved Models/lsm_model_90min_8_24_lambda0.t7');
model:get(2):remove(6);
model:get(2):remove(5);
model:get(2):remove(4);
--model:get(2):remove(3);
--model:get(2):remove(2);
model:evaluate()

baseline = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/Baseline_allTopo.t7')
present = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/90min_allTopo.t7')
local tot_subjects = baseline:size(1);
local exe_subjects = 12;
local con_subjects = 13;

baseline = baseline:resize(25*2500,3,64,64)
present = present:resize(25*2500,3,64,64)
Labels = torch.cat({torch.ones(13*2500),(2)*torch.ones(12*2500)})
SubjLabels = torch.linspace(1,25,25);
SubjLabels = torch.repeatTensor(SubjLabels,2500,1);
SubjLabels = SubjLabels:transpose(1,2):contiguous();
SubjLabels = SubjLabels:reshape(25*2500);
print('Data Loading done')

baseline:add(-baseline:mean())
baseline:div(baseline:std())
present:add(-present:mean())
present:div(present:std())
N=baseline:size(1)

featureTensor = torch.zeros(25*2500,8)

for i=1,N do
	local example1 = torch.Tensor(1,3,64,64);
    local example2 = torch.Tensor(1,3,64,64);
    example1[1] = baseline[i]
    example2[1] = present[i]
    local prediction = model:forward({example1:cuda(),example2:cuda()})
    featureTensor[i] = prediction:double();
end

local myFile = hdf5.open('Layer3.h5','w');
myFile:write('/home/features',featureTensor);
myFile:write('/home/Labels',Labels);
myFile:write('/home/SubjLabels',SubjLabels);