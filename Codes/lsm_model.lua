require 'nn';

model = nn.Sequential()
-- input: 1X64X55
model:add(nn.SpatialConvolution(1,6,5,1,2,1,2,0))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,1,2,1))

model:add(nn.SpatialConvolution(6,16,5,1,1,1,0,0))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,1,2,1))

model:add(nn.View(-1):setNumInputDims(3)) --13
model:add(nn.Linear(5120,1024)) --14
model:add(nn.ReLU())
model:add(nn.Linear(1024,512)) --14
model:add(nn.ReLU())

--]]
return model
