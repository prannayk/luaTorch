require 'crnn'

CharSplitLMMinibatchLoader = require 'crnn.util.CharSplitLMMinibatchLoader'
model_utils = require 'crnn.util.model_utils'
LSTM = require 'crnn.model.LSTM'

opt = {
    data_dir = '../datasets/crnn/tinyshakespeare',
    rnn_size = 128,
    num_layers = 10,
    model = 'lstm',
    learning_rate = 2e-3,
    learning_rate_decay = 0.97,
    learning_rate_decay_after = 10,
    decay_rate = 0.95,
    dropout = 0,
    seq_length = 100,
    batch_size = 50,
    max_epochs = 50,
    grad_clip = 5,
    train_frac = 0.85,
    valid_frac = 0.05,

    seed = 123,
    print_every = 10,
    eval_val_every = 100,
    checkpoint_dir = '../eval_runs/crnn_1',
    savefile = 'lstm',
    gpuid = -1
}
torch.setnumthreads(8)
torch.manualSeed(opt.seed)

test_frac = math.max(0,1- opt.train_frac - opt.valid_frac)
split_sizes = {opt.train_frac, opt.valid_frac, test_frac} 

print("Basic setup completed")

print(string.format('Train Split = %g\nValidation Split = %g\nTest Split = %g\n', unpack(split_sizes)))

print("Creating Data Interface")
loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
vocab_size = loader.vocab_size
print('Vocab Size: ' .. vocab_size)
if not path.exists(opt.checkpoint_dir) then
    lfs.mkdir(opt.checkpoint_dir)
end
protos = {}
print('creating an LSTM with' .. opt.num_layers .. 'layers')
protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    -- cell/hidden state zeroes
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end
protos.criterion = nn.ClassNLLCriterion()

params, grad_params = protos.rnn:getParameters()
params:uniform(-0.08,0.08)
print('number of parameters in the model: ' .. params:nElement())
-- builds LSTM layering
clones = {}
for name, proto in pairs(protos) do
    print('Cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length)
end

function eval_split(split_index, max_batches)
    print("Loss over: " .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end
    loader:reset_batch_pointer(split_index)
    local loss = 0
    local rnn_state = {[0] = init_state}
    for i = 1,n do
        local x,y = loader:next_batch(split_index)
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate()
            local lst = clones.rnn[t]:forward{x[{{},t}], unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            prediction = lst[#lst]
            loss = loss + clones.criterion[t]:forward(prediction, y[{{},t}])
        end
        rnn_state[0] = rnn_state[#rnn_state]
        print(i .. '/' .. n .. ': is running')
    end
    loss = loss / opt.seq_length / n
    return loss
end

local init_state_global = clone_list(init_state)
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    local x,y = loader:next_batch(1)
    local rnn_state = {[0] = init_state_global}
    local predictions = {}
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training()
        local lst = clones.rnn[t]:forward{x[{{},t}], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
        predictions[t] = lst[#lst]
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{},t}])
    end
    loss = loss / opt.seq_length
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)}
    for t=opt.seq_length,1,-1 do
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{},t}])
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x[{{},t}], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then
                drnn_state[t-1][k-1] = v
            end
        end
        init_state_global = rnn_state[#rnn_state]
        grad_params:clamp(-opt.grad_clip,opt.grad_clip)
        return loss, grad_params
    end
end
 
train_losses = {}
val_losses = {}
print(unpack(loader.split_sizes))
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i=1,iterations do
    local epoch = i / iterations_per_epoch
    local pepoch = math.floor(i/iterations_per_epoch) + 1
    local timer = torch.Timer()
    local _,loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real
    local train_loss = loss[1]
    print(pepoch .. ": " .. (i%iterations_per_epoch) .. "/".. iterations_per_epoch.. " : " .. train_loss)
    train_losses[i] = train_loss
    if i%iterations_per_epoch == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            optim_state.learningRate = optim_state.learning_rate * opt.learning_rate_decay
            print('Decayed rate')
        end
    end
    if i % opt.eval_val_every == 0 or i == iterations then
        local val_loss = eval_split(3)
        val_losses[i] = val_loss
        checkpoint = {}
        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('Saving at checkpoint')
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0*3 then
        print('Exploding, need to halt')
        break
    end
end
