require 'nn';
require 'paths';

print("Loading data")
if (not paths.filep("cifar10torchsmall.zip")) then
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
    os.execute('unzip cifar10torchsmall.zip')
end
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'}
print(trainset)
print(#trainset.data)
--itorch.image(trainset.data[100])
print(classes[trainset.label[100]])

-- data loaded

print("Data loaded")

-- setting up systematics
setmetatable(trainset, 
    {__index =  function(t,i)
                    return {t.data[i], t.label[i]}
                end}
);
trainset.data = trainset.data:double()
function trainset:size()
    return self.data:size(1)
end
print(trainset:size())
--itorch.image(trainset[33][1])
readChannel = trainset.data[{{},{1},{},{}}]
print(#readChannel)
print("Systematics done")
-- Build error things and error calculatins
mean = {}
stdv = {}
for i=1,3 do 
    mean[i] = trainset.data[{{},{i},{},{}}]:mean()
    trainset.data[{{},{i},{},{}}]:add(-mean[i])
    stdv[i] = trainset.data[{{},{i},{},{}}]:std()
    trainset.data[{{},{i},{},{}}]:div(stdv[i])
end


-- Building neural net
x = torch.rand(3,32,32)
print("Building neural net")
net = nn.Sequential()
net:add(nn.SpatialConvolution(3,6,5,5))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(6,12,5,5))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(12,16,5,5))
net:add(nn.SpatialConvolution(16,24,5,5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.ReLU())
net:add(nn.ReLU())
net:add(nn.View(24*8*8))
net:add(nn.Linear(24*8*8,480))
net:add(nn.ReLU())
net:add(nn.Linear(480,120))
net:add(nn.Linear(120,84))
net:add(nn.Linear(84,10))
net:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(net,criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 20
trainer:train(trainset)

-- neural network testing
testset.data = testset.data:double()
for i=1,3 do
    testset.data[{{},{i},{},{}}]:add(-mean[i])
    testset.data[{{},{i},{},{}}]:div(stdv[i])
end

correct = 0
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction,true)
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct/100 .. '%')

--end of story
print("done till this point")
