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
--trainPortion = 0.1

print("data loading done =====>")

indices = torch.randperm(baseline:size(1)):long()
testData_b = baseline:index(1,indices)
testData_p = present:index(1,indices)
testLabels = Labels:index(1,indices)

testData_b:add(-testData_b:mean())
testData_b:div(testData_b:std())
testData_p:add(-testData_p:mean())
testData_p:div(testData_p:std())
--print(trainData:mean(), trainData:std(), trainData:size(),trainLabels:size())
--torch.save('lsm_model.t7',model)
model = torch.load('lsm_model_90min.t7')
model:remove(3);
model2 = torch.load('lsm_model_30min.t7')
model:add(model2:get(3))


criterion = nn.ClassNLLCriterion()
N = testData_b:size(1)
batchSize = 500
teSize = N
print('Testing accuracy')
correct = 0
class_perform = {0,0}
class_size = {0,0}
classes = {'control','exercise'}
for i=1,N,batchSize do
    local groundtruth = testLabels:narrow(1,i,batchSize)
    example1 = testData_b:narrow(1,i,batchSize)
    example2 = testData_p:narrow(1,i,batchSize)
    local prediction = model:forward({example1,example2})
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    for j=1,batchSize do
        class_size[groundtruth[j]] = class_size[groundtruth[j]] +1
        if groundtruth[j] == indices[j][1] then
            correct = correct + 1
            class_perform[groundtruth[j]] = class_perform[groundtruth[j]] + 1
        end
    end
    collectgarbage()
end
print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
for i=1,#classes do
   print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
end
