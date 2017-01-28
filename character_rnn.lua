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
    print_every = 1,
    eval_val_every = 1000,
    checkpoint_dir = 'cv',
    savefile = 'lstm',
    gpuid = -1
}
torch.setnumthreads(4)
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
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end
protos.criterion = nn.ClassNLLCriterion()

params, grad_params = protos.rnn:getParameters()
params:uniform(-0.08,0.08)
print('number of parameters in the model: ' .. params:nElement())

clones = {}
for name, proto in pairs(protos) do
    print('Cloning' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length)
end
print(protos)
print(clones)
