require 'dp'
require 'optim'
ds = dp.Mnist()

trainInputs = ds:get('train', 'inputs', 'bchw')
trainTargets = ds:get('train', 'targets','b')
validInputs = ds:get('valid','inputs','bchw')
validTargets = ds:get('valid','targets','b')
print("Loaded data")
net = nn:Sequential()
net:add(nn.Convert('bchw','bf'))
net:add(nn.Linear(1*28*28,200))
net:add(nn.ReLU())
net:add(nn.Linear(200,80))
net:add(nn.Tanh())
net:add(nn.Linear(80,10))
net:add(nn.LogSoftMax())

-- criterion
criterion = nn.ClassNLLCriterion()

-- Validator
cm = optim.ConfusionMatrix(10)
function classEval(net, inputs, targets)
    cm:zero()
    for i=1,inputs:size(1) do
        local input,target = inputs[i], targets:narrow(1,i,1)
        local output = net:forward(input)
        cm:add(output,target)
    end
    cm:updateValids()
    return cm.totalValid
end

-- trainer : hand modelled, not plain SGD
function trainEpoch(net, criterion, input, target)
    for i=1,input:size(1) do
        if (i%1000 == 0) then io.write("=") end
        local input_this, target_this = input[i], target:narrow(1,i,1)
        local output_this = net:forward(input_this)
        local loss = criterion:forward(output_this, target_this)
        local gradOutput = criterion:backward(output_this,target_this)
        net:zeroGradParameters()
        local gradInput = net:backward(input_this, gradOutput)
        net:updateGradParameters(0.9)
        net:updateParameters(0.1)
    end
    io.write("|\n")
end


bestAccuracy, bestEpoch = 0,0
wait = 0
for epoch=1,300 do
    trainEpoch(net,criterion,trainInputs, trainTargets)
    local validAccuracy = classEval(net,validInputs, validTargets)
    if validAccuracy > bestAccuracy then
        bestAccuracy, bestEpoch = validAccuracy, epoch
        print(bestAccuracy, bestEpoch)
        wait = 0
    else
        wait = wait + 1
        print("Running: ", validAccuracy, epoch)
        if wait > 30 then break end
    end
end
