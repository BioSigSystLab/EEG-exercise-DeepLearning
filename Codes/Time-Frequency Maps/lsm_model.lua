require 'nn';

model = nn.Sequential()
-- input: 1X64X55
--[[model:add(nn.SpatialConvolution(1,6,5,1,2,1,2,0))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(6,16,3,1,1,1,1,0))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,1,2,1))

--model:add(nn.SpatialBatchNormalization(6))
model:add(nn.SpatialConvolution(16,32,3,1,1,1,1,0))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,1,2,1))
model:add(nn.SpatialConvolution(32,64,3,1,1,1,1,0))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,1,2,1))

--model:add(nn.SpatialBatchNormalization(32))
model:add(nn.SpatialConvolution(64,64,3,1,1,1,1,0))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,1,2,1))

model:add(nn.View(-1):setNumInputDims(3)) --13
model:add(nn.Linear(4096,1024)) --14
model:add(nn.ReLU())
model:add(nn.Linear(1024,256)) --14
model:add(nn.ReLU())

--]]

model:add(nn.SpatialConvolution(1,6,5,1,2,1,2,0))
--model:add(nn.SpatialBatchNormalization(6))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,1,2,1))
model:add(nn.SpatialConvolution(6,16,5,1,1,1,2,0))
--model:add(nn.SpatialBatchNormalization(16))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,1,2,1))
--model:add(nn.SpatialConvolution(16,32,3,1,1,1,1,0))
--model:add(nn.SpatialBatchNormalization(32))
--model:add(nn.ReLU())
--model:add(nn.SpatialMaxPooling(2,1,2,1))
--[[model:add(nn.View(-1):setNumInputDims(3)) --13
model:add(nn.Linear(4096,512)) --14
model:add(nn.ReLU())--]]
return model
