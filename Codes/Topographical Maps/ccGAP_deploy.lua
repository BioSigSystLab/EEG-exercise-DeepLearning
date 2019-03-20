require 'nn';
require 'image';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';
require 'hdf5';

--model = torch.load('Saved Models/lsm_model_90min.t7')
model = torch.load('lsm_model_90min.t7')
model:remove(2);
model:get(1):get(1):get(1):remove(9);
model:get(1):get(1):get(1):remove(6);
model:get(1):get(1):get(1):remove(3);
model:get(1):get(1):get(2):remove(9);
model:get(1):get(1):get(2):remove(6);
model:get(1):get(1):get(2):remove(3);
--model:get(1):remove(2)

inp = torch.rand(3,64,64):cuda();
out = model:forward({inp,inp})
--out = out:reshape(out:size(2),out:size(3),out:size(4))
numFilters = out:size(1)
print(numFilters)
print(#out)

baseline = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/Baseline_allTopo.t7')
present = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/90min_allTopo.t7')
local tot_subjects = baseline:size(1);
local exe_subjects = 12;
local con_subjects = 13;

baseline = baseline:resize(25*2500,3,64,64)
present = present:resize(25*2500,3,64,64)
--Labels = torch.cat({torch.ones(13*7500),(2)*torch.ones(12*7500)})
trainPortion = 1

print("data loading done =====>")

indices = torch.linspace(1,baseline:size(1),baseline:size(1)):long()
trainData_b = baseline:index(1,indices:sub(1,(trainPortion*indices:size(1))))
trainData_p = present:index(1,indices:sub(1,(trainPortion*indices:size(1))))
--trainLabels = Labels:index(1,indices:sub(1,(trainPortion*indices:size(1))))
indices = nil;

-- feature scaling here
trainData_b:add(-trainData_b:mean())
trainData_b:div(trainData_b:std())
trainData_p:add(-trainData_p:mean())
trainData_p:div(trainData_p:std())
-- feature scaling done

N=trainData_b:size(1)
trSize = trainData_b:size(1)

Weights = torch.load('ccGAP_errTensor.t7');
--Weights[{{},{1}}] = Weights[{{},{1}}] - Weights[numFilters+1][1];
--Weights[{{},{2}}] = Weights[{{},{2}}] - Weights[numFilters+1][2];

CAM = torch.zeros(25*2500,1,32,32);
for i=1,25 do
	for j=1,2500 do
		local inp1 = trainData_b[2500*(i-1)+j];
		local inp2 = trainData_p[2500*(i-1)+j];
		local out = model:forward({inp1:cuda(),inp2:cuda()});
		--out = out:reshape(out:size(2),out:size(3),out:size(4))
		--print(#out)
		if i<=13 then 
			W = Weights[{{},{1}}];
		else
			W = Weights[{{},{2}}];
		end
		--print(#W)
		local norm_const = W:sum();
		local A = W[1][1]*out[1]/norm_const;
		for f=2,numFilters do
			A = A+W[f][1]*out[f]/norm_const;
		end
		CAM[2500*(i-1)+j] = A:double();
	end
end
CON_CAM_file = hdf5.open('CON_ccCAM.h5','w');
CON_CAM_file:write('/home',CAM[{{1,13*2500},{},{},{}}])
CON_CAM_file:close()
EXE_CAM_file = hdf5.open('EXE_ccCAM.h5','w');
EXE_CAM_file:write('/home',CAM[{{13*2500+1,25*2500},{},{},{}}])
EXE_CAM_file:close()

CON_CAM = CAM[{{1,13*2500},{},{},{}}]:mean(1)
EXE_CAM = CAM[{{13*2500+1,25*2500},{},{},{}}]:mean(1)

CON_CAM:resize(CON_CAM:size(3),CON_CAM:size(4))
EXE_CAM:resize(EXE_CAM:size(3),EXE_CAM:size(4))
CON_CAM_img = torch.zeros(3,CON_CAM:size(1),CON_CAM:size(2))
EXE_CAM_img = torch.zeros(3,EXE_CAM:size(1),EXE_CAM:size(2))

CON_CAM_img[1] = torch.clamp(CON_CAM,0,CON_CAM:max())/CON_CAM:max();
CON_CAM_img[3] = torch.clamp(CON_CAM,CON_CAM:min(),0)/CON_CAM:min();

EXE_CAM_img[1] = torch.clamp(EXE_CAM,0,EXE_CAM:max())/EXE_CAM:max();
EXE_CAM_img[3] = torch.clamp(EXE_CAM,EXE_CAM:min(),0)/EXE_CAM:min();

--[[CON_CAM_img = torch.clamp(torch.abs(CON_CAM),0,torch.abs(CON_CAM):max())/torch.abs(CON_CAM):max();
EXE_CAM_img = torch.clamp(torch.abs(EXE_CAM),0,torch.abs(EXE_CAM):max())/torch.abs(EXE_CAM):max();--]]

image.save('CON_ccCAM.jpg',CON_CAM_img);
image.save('EXE_ccCAM.jpg',EXE_CAM_img);