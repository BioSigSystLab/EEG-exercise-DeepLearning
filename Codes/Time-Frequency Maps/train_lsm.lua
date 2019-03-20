require 'nn';
require 'image';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';
--local matio = require 'matio';

bmodel = require 'lsm_model.lua'
tmodel = require 'lsm_topmodel.lua'

model = nn.Sequential()
extractor = nn.Sequential()
p = nn.ParallelTable()
b2 = bmodel:clone('weight','bias','gradWeight','gradBias');
p:add(bmodel)
p:add(b2)
extractor:add(p)
--extractor:add(nn.JoinTable(2)) --1st dim is batchSize
extractor:add(nn.CSubTable())
model:add(extractor)
model:add(tmodel)
model:cuda()

SubjModel = nn.Sequential();
SubjModel:add(nn.View(-1):setNumInputDims(3))
SubjModel:add(nn.Dropout(0.5))
SubjModel:add(nn.Linear(7168,8));
--SubjModel:add(nn.BatchNormalization(64))
SubjModel:add(nn.ReLU());
--SubjModel:add(nn.Dropout(0.5))
--SubjModel:add(nn.Linear(64,32));
--SubjModel:add(nn.BatchNormalization(32))
--SubjModel:add(nn.ReLU());
SubjModel:add(nn.Linear(8,25));
SubjModel:add(nn.LogSoftMax());
SubjModel:cuda();


--baseline = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/torch_baseline.t7')
baseline = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/Baseline.t7')
--present = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/torch_min90.t7')
present = torch.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/EEG/90min.t7')
local tot_subjects = baseline:size(1);
local exe_subjects = 12;
local con_subjects = 13;

starting=2501; ending = 10000; --considering time only when task is presented --> 2501 is start of 1st second and 10k is end of 3 sec
--baseline = baseline[{{},{},{starting,ending},{}}]:transpose(2,3):contiguous():resize(25*7500,1,64,55)
--baseline = baseline:index(1,torch.linspace(10,25*7500,25*7500/10):long())
baseline = baseline:resize(25*7500,1,64,55)
--present = present[{{},{},{starting,ending},{}}]:transpose(2,3):contiguous():resize(25*7500,1,64,55)
--present = present:index(1,torch.linspace(10,25*7500,25*7500/10):long())
present = present:resize(25*7500,1,64,55)
Labels = torch.cat({torch.ones(13*7500),(2)*torch.ones(12*7500)})
SubjLabels = torch.linspace(1,25,25);
SubjLabels = torch.repeatTensor(SubjLabels,7500,1);
SubjLabels = SubjLabels:transpose(1,2):contiguous();
SubjLabels = SubjLabels:reshape(25*7500);
trainPortion = 0.8

print("data loading done =====>")

indices = torch.randperm(baseline:size(1)):long()
--indices = torch.cat({torch.linspace(1,0.4*baseline:size(1),0.4*baseline:size(1)),torch.linspace(0.6*baseline:size(1)+1,baseline:size(1),0.4*baseline:size(1)), torch.linspace(0.4*baseline:size(1)+1,0.6*baseline:size(1),0.2*baseline:size(1))}):long()
--indices = torch.cat({torch.linspace(1,0.48*baseline:size(1),0.48*baseline:size(1)),torch.linspace(0.56*baseline:size(1)+1,baseline:size(1),0.44*baseline:size(1)), torch.linspace(0.48*baseline:size(1)+1,0.56*baseline:size(1),0.08*baseline:size(1))}):long()
trainData_b = baseline:index(1,indices:sub(1,(trainPortion*indices:size(1))))
trainData_p = present:index(1,indices:sub(1,(trainPortion*indices:size(1))))
trainLabels = Labels:index(1,indices:sub(1,(trainPortion*indices:size(1))))
trainLabels_subj = SubjLabels:index(1,indices:sub(1,(trainPortion*indices:size(1))))
testData_b = baseline:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))
testData_p = present:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))
testLabels = Labels:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))
testLabels_subj = SubjLabels:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))
indices = nil;

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
N1 = testData_b:size(1)
local theta,gradTheta = model:getParameters()
local thetaAdv,gradThetaAdv = SubjModel:getParameters()
criterion = nn.ClassNLLCriterion():cuda()
--criterion = nn.MarginCriterion():cuda()
subj_criterion = nn.ClassNLLCriterion():cuda()
desired_subj_criterion = nn.DistKLDivCriterion():cuda()

local x1,x2,x3,x4,y,subj_y, subj_y2

local feval = function(params)
    if theta~=params then
        theta:copy(params)
    end
    gradTheta:zero()
    out = model:forward({x1,x2})
    --print(#out1,#out,#y)
    local loss = criterion:forward(out,y)
    local gradLoss = criterion:backward(out,y)
    model:backward({x1,x2},gradLoss)

    return loss, gradTheta
end

local subj_feval = function(params)
    if theta~=params then
        theta:copy(params)
    end
    gradTheta:zero()
    gradThetaAdv:zero()
    local input = {torch.cat(x1,x3,1),torch.cat(x2,x4,1)};
    local subj_y_tot = torch.cat(subj_y,subj_y2,1)
    out = extractor:forward(input)
    out1 = SubjModel:forward(out);

    advLoss = subj_criterion:forward(out1,subj_y_tot);
    local gradAdvLoss = subj_criterion:backward(out1,subj_y_tot);
    SubjModel:backward(extractor.output,gradAdvLoss);
    lambda = 0; --loss/advLoss;
    --local gradMinimax = SubjModel:updateGradInput(extractor.output, gradAdvLoss)--]]
    --extractor:backward(input,-lambda*gradMinimax);

    --[[local desired_subj_y = torch.zeros(out1:size()):cuda();
    for y_iter=1,out1:size(1) do
    	if subj_y_tot[y_iter]==1 then 
    		desired_subj_y[{{y_iter},{1,con_subjects}}]:fill(1.0/con_subjects)
    	else 
    		desired_subj_y[{{y_iter},{con_subjects+1,con_subjects+exe_subjects}}]:fill(1.0/exe_subjects)
    	end
    end--]]

    local desired_subj_y = torch.Tensor(out1:size()):fill(1.0/tot_subjects):cuda()
    local desired_advLoss = desired_subj_criterion:forward(out1,desired_subj_y);
    local desired_gradAdvLoss = desired_subj_criterion:backward(out1,desired_subj_y);
    local gradMinimax = SubjModel:updateGradInput(extractor.output, desired_gradAdvLoss)--]]
    extractor:backward(input,lambda*gradMinimax);

    return desired_advLoss, gradTheta
end

local advFeval = function(params)
    if thetaAdv~=params then
        thetaAdv:copy(params)
    end
    return advLoss, gradThetaAdv
end

batchSize = 100

indices = torch.randperm(trainData_b:size(1)):long()
trainData_b = trainData_b:index(1,indices)
trainData_p = trainData_p:index(1,indices)
trainLabels = trainLabels:index(1,indices)
trainLabels_subj = trainLabels_subj:index(1,indices)

mock_indices = torch.randperm(testData_b:size(1)):long()
mock_testData_b = testData_b:index(1,mock_indices)
mock_testData_p = testData_p:index(1,mock_indices)
mock_testLabels = testLabels:index(1,mock_indices)
mock_testLabels_subj = testLabels_subj:index(1,mock_indices)


epochs = 10
teAccuracy = 0
print('Training Starting')
local optimParams = {learningRate = 0.001, learningRateDecay = 0.0001, weightDecay = 0.05}
local adv_optimParams = {learningRate = 0.001, learningRateDecay = 0.0001, weightDecay = 0.05}
local _,loss
local losses = {}
local adv_losses = {}
local desired_adv_losses = {}
for epoch=1,epochs do
    collectgarbage()
    model:training()
    print('Epoch '..epoch..'/'..epochs)
    n2=1;
    for n=1,N-batchSize, batchSize do
        x1 = trainData_b:narrow(1,n,batchSize):cuda()
        x2 = trainData_p:narrow(1,n,batchSize):cuda()
        y = trainLabels:narrow(1,n,batchSize):cuda()
        subj_y = trainLabels_subj:narrow(1,n,batchSize):cuda()
        x3 = mock_testData_b:narrow(1,n2,torch.floor(batchSize*(1-trainPortion))):cuda()
        x4 = mock_testData_p:narrow(1,n2,torch.floor(batchSize*(1-trainPortion))):cuda()
        subj_y2 = mock_testLabels_subj:narrow(1,n2,torch.floor(batchSize*(1-trainPortion))):cuda()
        n2 = n2+torch.floor(batchSize*(1-trainPortion));
        if n2+batchSize*(1-trainPortion)>N1 then n2 = 1; end
        --print(y)
        _,loss = optim.adam(feval,theta,optimParams)
        losses[#losses + 1] = loss[1]
        _,loss = optim.adam(subj_feval,theta,optimParams)
        desired_adv_losses[#desired_adv_losses+1] = loss[1]; 
        _,loss = optim.adam(advFeval,thetaAdv,adv_optimParams)
        adv_losses[#adv_losses + 1] = loss[1]
    end
    local plots={{'Training Loss', torch.linspace(1,#losses,#losses), torch.Tensor(losses), '-'}}
    plots2={{'Adversary', torch.linspace(1,#adv_losses,#adv_losses), torch.Tensor(adv_losses), '-'}}
    plots3={{'Desired Adversary', torch.linspace(1,#desired_adv_losses,#desired_adv_losses), torch.Tensor(desired_adv_losses), '-'}}
    gnuplot.pngfigure('Training_vol_90.png')
    gnuplot.plot(table.unpack(plots))
    gnuplot.ylabel('Loss')
    gnuplot.xlabel('Batch #')
    gnuplot.plotflush()
    gnuplot.pngfigure('TrainingAdv_vol_90.png')
    gnuplot.plot(table.unpack(plots2))
    gnuplot.ylabel('Loss')
    gnuplot.xlabel('Batch #')
    gnuplot.plotflush()
    gnuplot.pngfigure('TrainingDesiredAdv_vol_90.png')
    gnuplot.plot(table.unpack(plots3))
    gnuplot.ylabel('Loss')
    gnuplot.xlabel('Batch #')
    gnuplot.plotflush()
    --permute training data
    indices = torch.randperm(trainData_b:size(1)):long()
    trainData_b = trainData_b:index(1,indices)
    trainData_p = trainData_p:index(1,indices)
    trainLabels = trainLabels:index(1,indices)
    trainLabels_subj = trainLabels_subj:index(1,indices)

    mock_indices = torch.randperm(testData_b:size(1)):long()
    mock_testData_b = testData_b:index(1,mock_indices)
    mock_testData_p = testData_p:index(1,mock_indices)
    mock_testLabels = testLabels:index(1,mock_indices)
    mock_testLabels_subj = testLabels_subj:index(1,mock_indices)

    if (epoch%2==0) then
    	model:evaluate()
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
            local prediction = model:forward({example1:cuda(),example2:cuda()})
            local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
            if groundtruth == indices[1][1] then
            --if testLabels[i]*prediction[1][1] > 0 then
                correct = correct + 1
                class_perform[groundtruth] = class_perform[groundtruth] + 1
            end
            collectgarbage()
        end
        print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
        if correct>=teAccuracy or epoch<=10 then
            teAccuracy=correct
            torch.save('lsm_model_90min.t7',model)
            for i=1,#classes do
               print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
            end
        end
    end
end

model = nil;
x1 = nil;
x2 = nil;
collectgarbage()
--torch.save('lsm_model.t7',model)
model = torch.load('lsm_model_90min.t7')
model:evaluate()

N = testData_b:size(1)
teSize = N
print('Testing accuracy')
correct = 0
class_perform = {0,0}
class_size = {0,0}
classes = {'control','exercise'}
for i=1,N do
    local groundtruth = testLabels[i]
    if groundtruth<0 then groundtruth=2 end
    local example1 = torch.Tensor(1,1,64,55);
    local example2 = torch.Tensor(1,1,64,55);
    example1[1] = testData_b[i]
    example2[1] = testData_p[i]
    class_size[groundtruth] = class_size[groundtruth] +1
    local prediction = model:forward({example1:cuda(),example2:cuda()})
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    --print(confidences)
    --print(#example1,#indices)
    --print('ground '..groundtruth, indices[1])
    if groundtruth == indices[1][1] then
    --if testLabels[i]*prediction[1][1] > 0 then
        correct = correct + 1
        class_perform[groundtruth] = class_perform[groundtruth] + 1
    end
    collectgarbage()
end
print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
for i=1,#classes do
   print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
end
