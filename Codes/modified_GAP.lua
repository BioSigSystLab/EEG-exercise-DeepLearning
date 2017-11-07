require 'nn';
require 'image';
require 'optim';
require 'gnuplot';

model = torch.load('lsm_model_90min.t7')

model2 = nn.Sequential()
model2:insert(model:get(3),1)
model:remove(3)
model2:insert(model:get(2),1)
model:remove(2)

p=nn.ParallelTable();
baseLinear = nn.Sequential();
baseLinear:insert(model:get(1):get(1):get(11),1)
model:get(1):get(1):remove(11)
baseLinear:insert(model:get(1):get(1):get(10),1)
model:get(1):get(1):remove(10)
baseLinear:insert(model:get(1):get(1):get(9),1)
model:get(1):get(1):remove(9)
baseLinear:insert(model:get(1):get(1):get(8),1)
model:get(1):get(1):remove(8)
baseLinear:insert(model:get(1):get(1):get(7),1)
model:get(1):get(1):remove(7)
p:add(baseLinear);
baseLinear2 = nn.Sequential();
baseLinear2:insert(model:get(1):get(2):get(11),1)
model:get(1):get(2):remove(11)
baseLinear2:insert(model:get(1):get(2):get(10),1)
model:get(1):get(2):remove(10)
baseLinear2:insert(model:get(1):get(2):get(9),1)
model:get(1):get(2):remove(9)
baseLinear2:insert(model:get(1):get(2):get(8),1)
model:get(1):get(2):remove(8)
baseLinear2:insert(model:get(1):get(2):get(7),1)
model:get(1):get(2):remove(7)
p:add(baseLinear2);
model2:insert(p,1);

print(model)
print(model2)

checkOut = model:forward({torch.rand(1,1,64,55),torch.rand(1,1,64,55)})
numFilters = checkOut[1]:size(3)
print(numFilters)
print(#checkOut[1])

baseline = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/torch_baseline.t7')
present = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/torch_min90.t7')

starting=2501; ending = 10000; --considering time only when task is presented --> 2501 is start of 1st second and 10k is end of 3 sec
baseline = baseline[{{},{},{starting,ending},{}}]:transpose(2,3):resize(25*7500,1,64,55)
present = present[{{},{},{starting,ending},{}}]:transpose(2,3):resize(25*7500,1,64,55)
Labels = torch.cat({torch.ones(13*7500),2*torch.ones(12*7500)})
trainPortion = 1

print("data loading done =====>")

indices = torch.randperm(baseline:size(1)):long()
trainData_b = baseline:index(1,indices:sub(1,(trainPortion*indices:size(1))))
trainData_p = present:index(1,indices:sub(1,(trainPortion*indices:size(1))))
trainLabels = Labels:index(1,indices:sub(1,(trainPortion*indices:size(1))))
trSize = trainData_b:size(1)

sorted,indices = torch.sort(trainLabels)
classes = {'control', 'exercise'}
classSize = torch.Tensor(#classes):zero()
for i=1,trSize do
	classSize[trainLabels[i]] = classSize[trainLabels[i]]+1
end
print(classSize)
print(classSize:sum())

criterion = nn.MSECriterion()

errorTensor = torch.Tensor(1+numFilters,#classes):zero()

batchSize = 500
indexDone=0
errorOccurred = 0

for c=1,#classes do
	print('Starting Class '..c)
	for t=1,classSize[c]/batchSize do
		local input1 = torch.Tensor(batchSize,1,64,55)
		local input2 = torch.Tensor(batchSize,1,64,55)
		local target = torch.Tensor(batchSize,2)
		classTarget = c
		for t1 = 1,batchSize do
			local t2 = (t-1)*batchSize + t1 + indexDone
			input1[t1] = trainData_b[indices[t2]]
			input2[t1] = trainData_p[indices[t2]]
			target[t1][trainLabels[indices[t2]]]=1;
			target[t1][3-trainLabels[indices[t2]]]=0;
			if target[t1][c]~=1 then
				print('error at c='..c..' t='..t..' t1='..t1)
				errorOccurred=1
				break;
			end
		end
		if errorOccurred==1 then 
			break 
		end
		local filters = model:forward({input1,input2})
		--print(filters)
		local error_ = criterion:forward(model2:forward(filters):exp(),target)*batchSize
		errorTensor[numFilters+1][classTarget] = errorTensor[numFilters+1][classTarget] + error_
		--print(errorTensor[numFilters+1][classTarget])
		for f=1,numFilters do
			collectgarbage()
			local filter_masked1 = torch.Tensor(filters[1]:size()):copy(filters[1])
			local filter_masked2 = torch.Tensor(filters[2]:size()):copy(filters[2])
			--print(filter_masked1[{{},{},{f},{}}]:sum(),filter_masked1[{{},{},{f+1},{}}]:sum())
			filter_masked1[{{},{},{f},{}}]:zero()
			filter_masked2[{{},{},{f},{}}]:zero()
			--print(filter_masked1[{{},{},{f},{}}]:sum(),filter_masked1[{{},{},{f+1},{}}]:sum())
			local error_masked = criterion:forward(model2:forward({filter_masked1,filter_masked2}):exp(),target)*batchSize
			errorTensor[f][classTarget] = errorTensor[f][classTarget] + error_masked
			--print(errorTensor[f][classTarget])
		end
	end
	if errorOccurred==1 then 
		break 
	end
	if classSize[c]%batchSize~=0 then
		local remaining = classSize[c]%batchSize
		local input1 = torch.Tensor(remaining,1,64,55)
		local input2 = torch.Tensor(remaining,1,64,55)
		local target = torch.Tensor(remaining,2)
		classTarget = c
		for t1 = 1,remaining do
			local t2 = classSize[c]-remaining + t1 + indexDone
			input1[t1] = trainData_b[indices[t2]]
			input2[t1] = trainData_p[indices[t2]]
			target[t1][trainLabels[indices[t2]]]=1;
			target[t1][3-trainLabels[indices[t2]]]=0;
			if target[t1][c]~=1 then
				print('error at c='..c..' t='..t..' t1='..t1)
				errorOccurred=1
				break;
			end
		end
		if errorOccurred==1 then 
			break 
		end
		local filters = model:forward({input1,input2})
		local error_ = criterion:forward(model2:forward(filters):exp(),target)*remaining
		errorTensor[numFilters+1][classTarget] = errorTensor[numFilters+1][classTarget] + error_
		for f=1,numFilters do
			collectgarbage()
			local filter_masked1 = torch.Tensor(filters[1]:size()):copy(filters[1])
			local filter_masked2 = torch.Tensor(filters[2]:size()):copy(filters[2])
			filter_masked1[{{},{},{f},{}}]:zero()
			filter_masked2[{{},{},{f},{}}]:zero()
			local error_masked = criterion:forward(model2:forward({filter_masked1,filter_masked2}):exp(),target)*remaining
			errorTensor[f][classTarget] = errorTensor[f][classTarget] + error_masked
		end
	end
	if errorOccurred==1 then 
		break 
	end
	indexDone = indexDone+classSize[c]
	--print(errorTensor)
end


torch.save('modifiedGAP_errTensor.t7',errorTensor)