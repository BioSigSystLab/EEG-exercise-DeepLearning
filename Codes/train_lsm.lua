require 'nn';
require 'image';
require 'optim';
require 'gnuplot';
--local matio = require 'matio';

bmodel = require 'lsm_model.lua'
tmodel = require 'lsm_topmodel.lua'

model = nn.Sequential()
p = nn.ParallelTable()
b2 = bmodel:clone('weight','bias','gradWeight','gradBias');
p:add(bmodel)
p:add(b2)
model:add(p)
model:add(nn.JoinTable(2)) --1st dim is batchSize
model:add(tmodel)


baseline = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/torch_baseline.t7')
present = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/torch_min30.t7')

starting=2501; ending = 10000; --considering time only when task is presented --> 2501 is start of 1st second and 10k is end of 3 sec
baseline = baseline[{{},{},{starting,ending},{}}]:transpose(2,3):resize(25*7500,1,64,55)
present = present[{{},{},{starting,ending},{}}]:transpose(2,3):resize(25*7500,1,64,55)
Labels = torch.cat({torch.ones(13*7500),2*torch.ones(12*7500)})
trainPortion = 0.8

print("data loading done =====>")

indices = torch.randperm(baseline:size(1)):long()
trainData_b = baseline:index(1,indices:sub(1,(trainPortion*indices:size(1))))
trainData_p = present:index(1,indices:sub(1,(trainPortion*indices:size(1))))
trainLabels = Labels:index(1,indices:sub(1,(trainPortion*indices:size(1))))
testData_b = baseline:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))
testData_p = present:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))
testLabels = Labels:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))

trainData_b:add(-trainData_b:mean())
trainData_b:div(trainData_b:std())
trainData_p:add(-trainData_p:mean())
trainData_p:div(trainData_p:std())

testData_b:add(-testData_b:mean())
testData_b:div(testData_b:std())
testData_p:add(-testData_p:mean())
testData_p:div(testData_p:std())
--print(trainData:mean(), trainData:std(), trainData:size(),trainLabels:size())
N=trainData_b:size(1)
local theta,gradTheta = model:getParameters()
criterion = nn.ClassNLLCriterion()

local x1,x2,y

local feval = function(params)
    if theta~=params then
        theta:copy(params)
    end
    gradTheta:zero()
    out = model:forward({x1,x2})
    --print(#x,#out,#y)
    local loss = criterion:forward(out,y)
    local gradLoss = criterion:backward(out,y)
    model:backward({x1,x2},gradLoss)
    return loss, gradTheta
end

batchSize = 1000

indices = torch.randperm(trainData_b:size(1)):long()
--trainData = trainData:index(1,indices)
--trainLabels = trainLabels:index(1,indices)

epochs = 50
teAccuracy = 0
print('Training Starting')
local optimParams = {learningRate = 0.002, learningRateDecay = 0.00001}
local _,loss
local losses = {}
for epoch=1,epochs do
    collectgarbage()
    print('Epoch '..epoch..'/'..epochs)
    for n=1,N-batchSize, batchSize do
        x1 = trainData_b:narrow(1,n,batchSize)
        x2 = trainData_p:narrow(1,n,batchSize)
        y = trainLabels:narrow(1,n,batchSize)
        --print(y)
        _,loss = optim.adam(feval,theta,optimParams)
        losses[#losses + 1] = loss[1]
        --print('Batch '.. n..' done')
    end
    if (epoch%5==1) then
        local plots={{'Training Loss', torch.linspace(1,#losses,#losses), torch.Tensor(losses), '-'}}
        gnuplot.pngfigure('Training_30.png')
        gnuplot.plot(table.unpack(plots))
        gnuplot.ylabel('Loss')
        gnuplot.xlabel('Batch #')
        gnuplot.plotflush()
    end
    --permute training data
    indices = torch.randperm(trainData_b:size(1)):long()
    trainData_b = trainData_b:index(1,indices)
    trainData_p = trainData_p:index(1,indices)
    trainLabels = trainLabels:index(1,indices)

    if (epoch%5==0) then
        N1 = testData_b:size(1)
        teSize = N1
        --print('Testing accuracy')
        correct = 0
        class_perform = {0,0}
        class_size = {0,0}
        classes = {'control','exercise'}
        for i=1,N1 do
            local groundtruth = testLabels[i]
            local example1 = torch.Tensor(1,1,64,55);
            local example2 = torch.Tensor(1,1,64,55);
            example1[1] = testData_b[i]
            example2[1] = testData_p[i]
            class_size[groundtruth] = class_size[groundtruth] +1
            local prediction = model:forward({example1,example2})
            local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
            if groundtruth == indices[1][1] then
                correct = correct + 1
                class_perform[groundtruth] = class_perform[groundtruth] + 1
            end
            collectgarbage()
        end
        print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
        if correct>teAccuracy then
            teAccuracy=correct
            torch.save('lsm_model_30min.t7',model)
            for i=1,#classes do
               print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
            end
        end
    end
end

--torch.save('lsm_model.t7',model)
model = torch.load('lsm_model_30min.t7')


N = testData_b:size(1)
teSize = N
print('Testing accuracy')
correct = 0
class_perform = {0,0}
class_size = {0,0}
classes = {'control','exercise'}
for i=1,N do
    local groundtruth = testLabels[i]
    local example1 = torch.Tensor(1,1,64,55);
    local example2 = torch.Tensor(1,1,64,55);
    example1[1] = testData_b[i]
    example2[1] = testData_p[i]
    class_size[groundtruth] = class_size[groundtruth] +1
    local prediction = model:forward({example1,example2})
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    --print(confidences)
    --print(#example1,#indices)
    --print('ground '..groundtruth, indices[1])
    if groundtruth == indices[1][1] then
        correct = correct + 1
        class_perform[groundtruth] = class_perform[groundtruth] + 1
    end
    collectgarbage()
end
print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
for i=1,#classes do
   print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
end
