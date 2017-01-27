require 'dp'
require 'nn'
require 'optim'

ds = dp.Mnist()

trainInput = ds:get('train','input','bchw')
trainTarget = ds:get('train', 'target', 'b')
validInput = ds:get('valid','input','bchw')
validTarget = ds:get('valid','target','b')

net = nn.Sequential()
net:add(nn.SpatialConvolution(1,6,3,3))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(6,16,3,3))
net:add(nn.Tanh())
net:add(nn.SpatialMaxPooling(2,2))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(16,24,5,5))
net:add(nn.SpatialMaxPooling(2,2))
net:add(nn.View(24*4*4))
net:add(nn.Linear(24*4*4,480))
net:add(nn.Linear(480,120))
net:add(nn.Tanh())
net:add(nn.Linear(120,64))
net:add(nn.Tanh())
net:add(nn.Linear(64,10))
net:add(nn.LogSoftMax())

--criterion
criterion = nn.ClassNLLCriterion()

-- validator
cm = optim.ConfusionMatrix(100)
function evalValidset(net, inputs, targets)
    cm:zero()
    for i=1,inputs:size(1) do
        local input,target = intputs[i], targets:narrow(1,i,1)
        local output = net:forward(input)
        cm:add(output,target)
    end
    cm:updateValids()
    return cm.totalValid
end

--trainer
function runEpoch(net, inputs, targets, criterion)
    for i=1,inputs:size(1) do
        local input,target = input[i], target:narrow(1,i,1)
        local output = net:forward(input)
        net:zeroGradParameters()
        local loss = net:forward(output,target)
        local gradOutput = net:backward(output,target)
        local gradInput = net:backward(input,gradOutput)
        net:updateGradParameters(0.9)
        net:updateParameters(0.1)
    end
end

-- trainer run
function learnMNIST()
    bestAcc, bestEp = 0,0 
    wait = 0
    for i=1,300 do 
        runEpoch(net, inputs, targets, criterion)
        local validAcc = evalValidset(net,validInputs, validTargets)
        if validAcc > bestAcc then
            bestAcc, bestEp = validAcc,i
            print(bestAcc,bestEp,": New Maxima reached")
            wait = 0
        else
            print(validAcc,": Not realliy doing better")
            wait = wait + 1
            if wait > 30 then print("Too much, exiting") end
        end
    end
end
