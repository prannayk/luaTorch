require 'dp'
require 'nn'
require 'optim'

ds = dp.Mnist()

trainInput = ds:get('train','input','bchw')
trainTarget = ds:get('train', 'target', 'b')
validInput = ds:get('valid','input','bchw')
validTarget = ds:get('valid','target','b')

trainInput = trainInput:type('torch.DoubleTensor')
validInput = validInput:type('torch.DoubleTensor')

net = nn.Sequential()
--net:add(nn.Convert('bhwc','bchw'))
net:add(nn.SpatialConvolution(1,6,3,3))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(6,16,3,3))
net:add(nn.ReLU())
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
cm = optim.ConfusionMatrix(10)
function evalValidset(net, inputs, targets)
    cm:zero()
    for i=1,inputs:size(1) do
        local input,target = inputs[i], targets:narrow(1,i,1)
        local output = net:forward(input)
        cm:add(output,target)
    end
    cm:updateValids()
    return cm.totalValid
end

--trainer
function runEpoch(net, inputs, targets, criterion)
    for id=1,inputs:size(1) do
        local i=math.random(1,inputs:size(1))
        if ((id%1000) == 0) then print(id) end
        if((id%20000) == 0) then break end
        local input,target = inputs[i], targets:narrow(1,i,1)
        local output = net:forward(input)
        net:zeroGradParameters()
        local loss = criterion:forward(output,target)
        local gradOutput = criterion:backward(output,target)
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
        runEpoch(net, trainInput, trainTarget, criterion)
        local validAcc = evalValidset(net,validInput, validTarget)
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
learnMNIST()
--cnn = nn.Sequential()
--cnn:add(nn.Convert('bhwc','bchw'))
--cnn:add(nn.SpatialConvolution(1,16,5,5,1,1,2,2))
--cnn:add(nn.ReLU())
--cnn:add(nn.SpatialMaxPooling(2,2,2,2))
--cnn:add(nn.SpatialConvolution(16,32,5,5,1,1,2,2))
--cnn:add(nn.ReLU())
--cnn:add(nn.SpatialMaxPooling(2,2,2,2))
--outsize = cnn:outside{1,28,28,1}
--print("here")
--cnn:add(nn.Collapse(3))
--cnn:add(nn.Linear(outsize[2]*outsize[3]*outsize[4],200))
--cnn:add(nn.ReLU())
--cnn:add(nn.Linear(200,10))
--cnn:add(nn.LogSoftMax())
--print("here")
--train = dp.Optimizer{
--    loss = nn.ModuleCriterion(nn.ClassNLLCriterion,nil,nn.Convert()),
--    callback = function(model,report)
--        model:updateGradParameters(0.9)
--        model:updateParameters(0.1)
--        model:maxParamNorm(2)
--        model:zeroGradParameters()
--    end,
--    feedback = dp.Confusion(),
--    sampler = dp.ShuffleSampler{batch_size = 32},
--    progress = true
--}
--valid = dp.Evaluator{
---    feedback = dp.Confusion(), sampler = dp.Sampler{batch_size=32}
--}
--test = dp.Evaluator{
--    feedback = dp.Confusion(), sampler = dp.Sampler{batch_size = 32}
--}
--xp = dp.Experiment{
--    model = cnn,
--    optimizer = train, validator = valid, tester = test,
--    observer = dp.EarlyStopper{
--        error_report = {'validator','feedback','confusion','accuracy'},
--        maximize = true, max_epochs = 50
--    },
--    random_seed = os.time(), max_epoch=2000
--}
--require 'cutorch'
--require 'cunn'
--print(cnn)
--xp:cuda()
--xp:run(ds)
