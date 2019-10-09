require 'nn';
require 'image';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';
require 'hdf5';

model = torch.load('Saved Models/lsm_model_90min.t7')

model:remove(2);
--model:get(1):get(1):get(1):remove(9);
model:get(1):get(1):get(1):remove(6);
model:get(1):get(1):get(1):remove(3);
--model:get(1):get(1):get(2):remove(9);
model:get(1):get(1):get(2):remove(6);
model:get(1):get(1):get(2):remove(3);
--model:get(1):remove(2)

inp = torch.rand(1,64,55):cuda();
out = model:forward({inp,inp})
numFilters = out:size(1)
electrodes = out:size(2)
print(numFilters)
print(electrodes)
print(#out)

baseline = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/Baseline.t7')
present = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/90min.t7')
--print(present[{{},{},{},{50,55}}]:sum())
local tot_subjects = baseline:size(1);
local exe_subjects = 12;
local con_subjects = 13;

baseline = baseline:resize(25*7500,1,64,55)
present = present:resize(25*7500,1,64,55)
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

W_model = torch.load('GAP_Weights.t7');
W_model = W_model:get(3);
Weights = W_model:parameters();
Weights = Weights[1];

CAM = torch.zeros(25*7500,1,28);
for i=1,25 do
	for j=1,7500 do
		local inp1 = trainData_b[7500*(i-1)+j];
		local inp2 = trainData_p[7500*(i-1)+j];
		local out = model:forward({inp1:cuda(),inp2:cuda()});
		--print(#out)
		if i<=13 then 
			W = Weights[1];
		else
			W = Weights[2];
		end
		--print(#W)
		local A = torch.zeros(1,28):cuda();
		for f=1,numFilters do
			for e=1,electrodes do
				A = A+(W[electrodes*(f-1)+e]*out[f][e]):resize(1,28);
			end
		end
		CAM[7500*(i-1)+j] = A:double();
	end
	print(i)
end
CON_CAM_file = hdf5.open('CON_CAM_GAP.h5','w');
CON_CAM_file:write('/home',CAM[{{1,13*7500},{},{}}])
CON_CAM_file:close()
EXE_CAM_file = hdf5.open('EXE_CAM_GAP.h5','w');
EXE_CAM_file:write('/home',CAM[{{13*7500+1,25*7500},{},{}}])
EXE_CAM_file:close()