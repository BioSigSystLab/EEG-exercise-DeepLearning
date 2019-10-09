require 'nn';
require 'image';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';

model = torch.load('Saved Models/lsm_model_90min.t7')

model:remove(2);
inp = torch.rand(1,1,64,55):cuda();
out = model:forward({inp,inp})
--model:get(1):get(1):get(1):add(nn.SpatialAveragePooling(out:size(4),out:size(3)))
--model:get(1):get(1):get(2):add(nn.SpatialAveragePooling(out:size(4),out:size(3)))
--model = model:get(1)
model:cuda()
model2 = nn.Sequential()
model2:add(nn.SpatialAveragePooling(out:size(4),1))
model2:add(nn.View(-1):setNumInputDims(3))
model2:add(nn.Linear(16*64,2))
model2:add(nn.LogSoftMax())
model2:cuda()

print(model)
print(model2)

model:evaluate()
model2:training()

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
trainPortion = 0.8

print("data loading done =====>")

indices = torch.randperm(baseline:size(1)):long()
trainData_b = baseline:index(1,indices:sub(1,(trainPortion*indices:size(1))))
trainData_p = present:index(1,indices:sub(1,(trainPortion*indices:size(1))))
trainLabels = Labels:index(1,indices:sub(1,(trainPortion*indices:size(1))))
testData_b = baseline:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))
testData_p = present:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))
testLabels = Labels:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))
indices = nil;

-- feature scaling here
trainData_b:add(-trainData_b:mean())
trainData_b:div(trainData_b:std())
trainData_p:add(-trainData_p:mean())
trainData_p:div(trainData_p:std())

testData_b:add(-testData_b:mean())
testData_b:div(testData_b:std())
testData_p:add(-testData_p:mean())
testData_p:div(testData_p:std())
-- feature scaling done

N=trainData_b:size(1)
N1 = testData_b:size(1)

criterion = nn.ClassNLLCriterion():cuda()

local theta,gradTheta = model2:getParameters()
local x1,x2,y

local feval = function(params)
    if theta~=params then
        theta:copy(params)
    end
    gradTheta:zero()
    local out = model:forward({x1,x2})
    local out1 = model2:forward(out);
    --print(#out1,#out,#y)
    local loss = criterion:forward(out1,y)
    local gradLoss = criterion:backward(out1,y)
    model2:backward(out,gradLoss)

    return loss, gradTheta
end

batchSize = 100

indices = torch.randperm(trainData_b:size(1)):long()
trainData_b = trainData_b:index(1,indices)
trainData_p = trainData_p:index(1,indices)
trainLabels = trainLabels:index(1,indices)

epochs = 200
teAccuracy = 0
print('Training Starting')
local optimParams = {learningRate = 0.02, learningRateDecay = 0.001, weightDecay = 0.001}
local _,loss
local losses = {}

for epoch=1,epochs do
    collectgarbage()
    --model:training()
    model2:training()
    print('Epoch '..epoch..'/'..epochs)
    for n=1,N-batchSize, batchSize do
        x1 = trainData_b:narrow(1,n,batchSize):cuda()
        x2 = trainData_p:narrow(1,n,batchSize):cuda()
        y = trainLabels:narrow(1,n,batchSize):cuda()
        --print(y)
        _,loss = optim.sgd(feval,theta,optimParams)
        losses[#losses + 1] = loss[1]
    end
    local plots={{'Training Loss', torch.linspace(1,#losses,#losses), torch.Tensor(losses), '-'}}
    gnuplot.pngfigure('Training_GAP_90.png')
    gnuplot.plot(table.unpack(plots))
    gnuplot.ylabel('Loss')
    gnuplot.xlabel('Batch #')
    gnuplot.plotflush()
    --permute training data
    indices = torch.randperm(trainData_b:size(1)):long()
    trainData_b = trainData_b:index(1,indices)
    trainData_p = trainData_p:index(1,indices)
    trainLabels = trainLabels:index(1,indices)


    if (epoch%10==0) then
    	model:evaluate()
    	model2:evaluate()
        N1 = testData_b:size(1)
        teSize = N1
        --print('Testing accuracy')
        correct = 0
        class_perform = {0,0}
        class_size = {0,0}
        classes = {'control','exercise'}
        for i=1,N1 do
            local groundtruth = testLabels[i]
            if groundtruth<0 then groundtruth=2 end
            local example1 = torch.Tensor(1,1,64,55);
            local example2 = torch.Tensor(1,1,64,55);
            example1[1] = testData_b[i]
            example2[1] = testData_p[i]
            class_size[groundtruth] = class_size[groundtruth] +1
            local prediction = model2:forward(model:forward({example1:cuda(),example2:cuda()}));
            local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
            if groundtruth == indices[1][1] then
            --if testLabels[i]*prediction[1][1] > 0 then
                correct = correct + 1
                class_perform[groundtruth] = class_perform[groundtruth] + 1
            end
            collectgarbage()
        end
        print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
        if correct>=teAccuracy then
            teAccuracy=correct
            torch.save('GAP_Weights.t7',model2)
            for i=1,#classes do
               print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
            end
        end
    end
end

