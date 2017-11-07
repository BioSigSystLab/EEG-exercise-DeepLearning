require 'nn';
require 'image';
require 'optim';
require 'gnuplot';
local matio = require 'matio';

model = require 'model.lua'

train = matio.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/EEG_dat.mat')

Data = train.A:contiguous()
Labels = train.B:resize(25):contiguous()
trainPortion = 0.8

indices = torch.randperm(Data:size(1)):long()
trainData = Data:index(1,indices:sub(1,(trainPortion*indices:size(1))))
trainLabels = Labels:index(1,indices:sub(1,(trainPortion*indices:size(1))))
testData = Data:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))
testLabels = Labels:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))

trainData:add(-trainData:mean())
trainData:div(trainData:std())
--print(trainData:mean(), trainData:std(), trainData:size(),trainLabels:size())
N=trainData:size(1)
local theta,gradTheta = model:getParameters()
criterion = nn.ClassNLLCriterion()

local x,y

local feval = function(params)
    if theta~=params then
        theta:copy(params)
    end
    gradTheta:zero()
    out = model:forward(x)
    --print(#x,#out,#y)
    local loss = criterion:forward(out,y)
    local gradLoss = criterion:backward(out,y)
    model:backward(x,gradLoss)
    return loss, gradTheta
end

batchSize = 20

indices = torch.randperm(trainData:size(1)):long()
--trainData = trainData:index(1,indices)
--trainLabels = trainLabels:index(1,indices)

epochs = 20
teAccuracy = 0
print('Training Starting')
local optimParams = {learningRate = 0.001, learningRateDecay = 0.0001}
local _,loss
local losses = {}
for epoch=1,epochs do
    collectgarbage()
    print('Epoch '..epoch..'/'..epochs)
    for n=1,N, batchSize do
        x = trainData:narrow(1,n,batchSize)
        y = trainLabels:narrow(1,n,batchSize)
        --print(y)
        _,loss = optim.adam(feval,theta,optimParams)
        losses[#losses + 1] = loss[1]
    end
    local plots={{'Training Loss', torch.linspace(1,#losses,#losses), torch.Tensor(losses), '-'}}
    gnuplot.pngfigure('Training2.png')
    gnuplot.plot(table.unpack(plots))
    gnuplot.ylabel('Loss')
    gnuplot.xlabel('Batch #')
    gnuplot.plotflush()
    --permute training data
    indices = torch.randperm(trainData:size(1)):long()
    trainData = trainData:index(1,indices)
    trainLabels = trainLabels:index(1,indices)


    N = testData:size(1)
    teSize = N
    --print('Testing accuracy')
    correct = 0
    class_perform = {0,0}
    class_size = {0,0}
    classes = {'exercise','control'}
    for i=1,N do
        local groundtruth = testLabels[i]
        local example = torch.Tensor(2049,1);
        example = testData[i]
        class_size[groundtruth] = class_size[groundtruth] +1
        local prediction = model:forward(example)
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        --print(#example,#indices)
        --print('ground '..groundtruth, indices[1])
        if groundtruth == indices[1] then
            correct = correct + 1
            class_perform[groundtruth] = class_perform[groundtruth] + 1
        end
        collectgarbage()
    end
    print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
    if correct>teAccuracy and epoch>5 then
        teAccuracy=correct
        torch.save('linear.t7',model)
        for i=1,#classes do
           print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
        end
    end
end

model = torch.load('linear.t7')
N = testData:size(1)
teSize = N
print('Testing accuracy')
correct = 0
class_perform = {0,0}
class_size = {0,0}
classes = {'exercise','control'}
for i=1,N do
    local groundtruth = testLabels[i]
    local example = torch.Tensor(2049,1);
    example = testData[i]
    class_size[groundtruth] = class_size[groundtruth] +1
    local prediction = model:forward(example)
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    --print(#example,#indices)
    --print('ground '..groundtruth, indices[1])
    if groundtruth == indices[1] then
        correct = correct + 1
        class_perform[groundtruth] = class_perform[groundtruth] + 1
    end
    collectgarbage()
end
print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
for i=1,#classes do
   print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
end