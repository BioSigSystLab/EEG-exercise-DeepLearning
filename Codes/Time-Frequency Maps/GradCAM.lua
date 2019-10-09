require 'nn';
require 'image';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';
require 'hdf5';

--model = torch.load('Saved Models/lsm_model_90min.t7')
model = torch.load('lsm_model_90min.t7')

model2 = model:get(2)
model:remove(2);
inp = torch.rand(1,1,64,55):cuda();
out = model:forward({inp,inp});
numFilters = out:size(2)
electrodes = out:size(3)
filterSize = out:size(4)
print(numFilters)
print(electrodes)
print(#out)

model:cuda()
model2:cuda()
print(model)
print(model2)

model:evaluate()
model2:evaluate()

baseline = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/Baseline.t7')
present = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/90min.t7')
local tot_subjects = baseline:size(1);
local exe_subjects = 12;
local con_subjects = 13;

baseline = baseline:resize(25*7500,1,64,55)
present = present:resize(25*7500,1,64,55)
Labels = torch.cat({torch.ones(13*7500),(2)*torch.ones(12*7500)})
trainPortion = 1

print("data loading done =====>")

indices = torch.linspace(1,baseline:size(1),baseline:size(1)):long()
trainData_b = baseline:index(1,indices:sub(1,(trainPortion*indices:size(1))))
trainData_p = present:index(1,indices:sub(1,(trainPortion*indices:size(1))))
trainLabels = Labels:index(1,indices:sub(1,(trainPortion*indices:size(1))))
indices = nil;

-- feature scaling here
trainData_b:add(-trainData_b:mean())
trainData_b:div(trainData_b:std())
trainData_p:add(-trainData_p:mean())
trainData_p:div(trainData_p:std())
-- feature scaling done

N = trainData_b:size(1)
trSize = trainData_b:size(1)

criterion = nn.ClassNLLCriterion():cuda()
local theta,gradTheta = model2:getParameters()

CAM = torch.zeros(25*7500,1,filterSize);

for i=1,25 do
    for j=1,7500 do
        gradTheta:zero()
        local inp1 = trainData_b[7500*(i-1)+j];
        local inp2 = trainData_p[7500*(i-1)+j];
        local y = trainLabels[7500*(i-1)+j];
        local out = model:forward({inp1:cuda(),inp2:cuda()});
        --print(#out)
        local out1 = model2:forward(out);
        local loss = criterion:forward(out1,y);
        local gradLoss = criterion:backward(out1,y);
        local Weights = model2:updateGradInput(out, gradLoss);
        --print(#Weights)
        local W = Weights:mean(3); -- size of W = numFilters X electrodes X 1
        --W = W:mean(2); -- average over the electrodes to get one weight value per filter
        --print(#W)
        --print(W[2][1][1])
        local A = torch.zeros(electrodes,filterSize):cuda();
        for f=1,numFilters do
            for e=1,electrodes do     -- uncomment if separate weights for electrodes is desired
                A[e] = A[e]+(W[f][e][1]*out[f][e]):resize(filterSize); 
            end
        end
        CAM[7500*(i-1)+j] = (A:mean(1)):double();
    end
    print(i)
end
CON_CAM_file = hdf5.open('CON_CAM_grad.h5','w');
CON_CAM_file:write('/home',CAM[{{1,13*7500},{},{}}])
CON_CAM_file:close()
EXE_CAM_file = hdf5.open('EXE_CAM_grad.h5','w');
EXE_CAM_file:write('/home',CAM[{{13*7500+1,25*7500},{},{}}])
EXE_CAM_file:close()