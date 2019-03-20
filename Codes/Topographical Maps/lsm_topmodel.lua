require 'nn';

model = nn.Sequential()

model:add(nn.View(-1):setNumInputDims(3))
--model:add(nn.Dropout(0.5))
--model:add(nn.BatchNormalization(1536))
--model:add(nn.Linear(512,512))
--model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(1024,8))
--model:add(nn.BatchNormalization(64))
model:add(nn.ReLU())
--model:add(nn.Dropout(0.5))
--model:add(nn.Linear(64,8))
--model:add(nn.BatchNormalization(8))
--model:add(nn.ReLU())
--model:add(nn.Dropout(0.5))
model:add(nn.Linear(8,2))
--model:add(nn.Tanh())
model:add(nn.LogSoftMax())

return model