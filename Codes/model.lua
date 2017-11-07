require 'nn';

model = nn.Sequential()

model:add(nn.View(-1):setNumInputDims(2)) --13
model:add(nn.Linear(320,500)) --14
model:add(nn.ReLU()) --15
model:add(nn.Dropout(0.5)) --16
model:add(nn.Linear(500,50)) --17
model:add(nn.ReLU()) --18
model:add(nn.Dropout(0.5)) --19
model:add(nn.Linear(50,2)) --20
model:add(nn.LogSoftMax())-- 21
--]]
return model
