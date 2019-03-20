require 'nn';
require 'image';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';

model = torch.load('Saved Models/lsm_model_90min.t7')
--model = torch.load('lsm_model_90min.t7')

model2 = nn.Sequential()
model2:add(model:get(2))
model:remove(2);

model:cuda()
model2:cuda()
model:evaluate()
model2:evaluate()

print(model)
print(model2)

inp = torch.rand(1,64,55):cuda();
out = model:forward({inp,inp})
numFilters = out:size(1)
print(numFilters)
print(#out)

--baseline = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/torch_baseline.t7')
baseline = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/Baseline.t7')
--present = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/torch_min90.t7')
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

N=trainData_b:size(1)
trSize = trainData_b:size(1)

sorted,indices = torch.sort(trainLabels)
classes = {'control', 'exercise'}
classSize = torch.Tensor(#classes):zero()
for i=1,trSize do
    classSize[trainLabels[i]] = classSize[trainLabels[i]]+1
end
print(classSize)
print(classSize:sum())

criterion = nn.ClassNLLCriterion():cuda()

WeightsTensor = torch.zeros(numFilters,#classes)

batchSize = 100
indexDone=0
errorOccurred = 0


for c=1,#classes do
    print('Starting Class '..c)
    for t=1,classSize[c]/batchSize do
        local input1 = torch.CudaTensor(batchSize,1,64,55)
        local input2 = torch.CudaTensor(batchSize,1,64,55)
        local target = torch.CudaTensor(batchSize)
        classTarget = c
        for t1 = 1,batchSize do
            local t2 = (t-1)*batchSize + t1 + indexDone
            input1[t1] = trainData_b[indices[t2]]
            input2[t1] = trainData_p[indices[t2]]
            target[t1] = trainLabels[indices[t2]];
            --target[t1][trainLabels[indices[t2]]]=1;
            --target[t1][3-trainLabels[indices[t2]]]=0;
            if target[t1]~=c then
                print('error at c='..c..' t='..t..' t1='..t1)
                errorOccurred=1
                break;
            end
        end
        if errorOccurred==1 then 
            break 
        end
        local filters = model:forward({input1,input2})
        --print(#filters)
        for t1=1,batchSize do
            local error_ = criterion:forward(model2:forward(filters[t1]),target[t1])
            for f=1,numFilters do
                collectgarbage()
                local filter_masked = torch.CudaTensor(filters[t1]:size()):copy(filters[t1])
                --print(#filter_masked)
                --print(filter_masked1[{{},{},{f},{}}]:sum(),filter_masked1[{{},{},{f+1},{}}]:sum())
                filter_masked[{{f},{},{}}]:zero()
                --print(filter_masked1[{{},{},{f},{}}]:sum(),filter_masked1[{{},{},{f+1},{}}]:sum())
                local error_masked = criterion:forward(model2:forward(filter_masked),target[t1])
                if (torch.pow(filters[t1][f],2):sum())==0 and (error_masked-error_)~=0 then print('Bug in Filter '..f..' for example c='..c..' t='..t..' t1='..t1) end
                if (torch.pow(filters[t1][f],2):sum())~=0 then WeightsTensor[f][classTarget] = WeightsTensor[f][classTarget] + (error_masked-error_)/(torch.pow(filters[t1][f],2):sum()); end
                --print(errorTensor[f][classTarget])
            end
        end
    end
    if errorOccurred==1 then 
        break 
    end
    if classSize[c]%batchSize~=0 then
        local remaining = classSize[c]%batchSize
        local input1 = torch.Tensor(remaining,1,64,55)
        local input2 = torch.Tensor(remaining,1,64,55)
        local target = torch.Tensor(remaining)
        classTarget = c
        for t1 = 1,remaining do
            local t2 = classSize[c]-remaining + t1 + indexDone
            input1[t1] = trainData_b[indices[t2]]
            input2[t1] = trainData_p[indices[t2]]
            target[t1] = trainLabels[indices[t2]]
            --target[t1][trainLabels[indices[t2]]]=1;
            --target[t1][3-trainLabels[indices[t2]]]=0;
            if target[t1]~=c then
                print('error at c='..c..' t='..t..' t1='..t1)
                errorOccurred=1
                break;
            end
        end
        if errorOccurred==1 then 
            break 
        end
        local filters = model:forward({input1,input2})
        for t1=1,remaining do
            local error_ = criterion:forward(model2:forward(filters[t1]),target[t1])
            for f=1,numFilters do
                collectgarbage()
                local filter_masked = torch.CudaTensor(filters[t1]:size()):copy(filters[t1])
                filter_masked[{{f},{},{}}]:zero()
                local error_masked = criterion:forward(model2:forward(filter_masked),target[t1])
                if (torch.pow(filters[t1][f],2):sum())==0 and (error_masked-error_)~=0 then print('Bug in Filter '..f..' for example c='..c..' t='..t..' t1='..t1) end
                if (torch.pow(filters[t1][f],2):sum())~=0 then WeightsTensor[f][classTarget] = WeightsTensor[f][classTarget] + (error_masked-error_)/(torch.pow(filters[t1][f],2):sum()); end
            end
        end
    end
    if errorOccurred==1 then 
        break 
    end
    indexDone = indexDone+classSize[c]
    --print(errorTensor)
end
torch.save('ccGAP_errTensor1.t7',WeightsTensor)