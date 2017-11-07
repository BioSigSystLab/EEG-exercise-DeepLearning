require 'nn';

model = nn.Sequential()

model:add(nn.View(-1):setNumInputDims(1))
model:add(nn.Linear(1024,128))
model:add(nn.ReLU())
model:add(nn.Linear(128,16))
model:add(nn.ReLU())
model:add(nn.Linear(16,2))
model:add(nn.LogSoftMax())

return model