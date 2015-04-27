stringx = require('pl.stringx')
require 'io'
require 'cunn'
cutorch.setDevice(1)
require 'nngraph'
require('base')

params = {batch_size=100,
		  seq_length=400,
		  layers=2,
		  rnn_size=400,
		  vocab_size=50}
vocabName = 'vocabChar.t7b'
vocab = torch.load(vocabName)

charCoreNetName = 'charCoreNet.t7b'
charCoreNet = torch.load(charCoreNetName)
function transfer_data(x)
  return x:cuda()
end

function setup()
  local core_network = charCoreNet
  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
end

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if #line ~= 1 then error({code="more"}) end
  if vocab[line[1]] == nil then error({code="vocab", word = line[1]}) end
  return line
end

model = {}
setup()
-- not really needed but I left it for future implementations
g_disable_dropout(model.rnns)

--[[
io.write("\nOK GO\n")
io.flush()
while true do
  local ok, line = pcall(readline)
  if not ok then
    if line.code == "EOF" then
      break -- end loop
    elseif line.code == "more" then
      io.write("Only one character per line\n")
      io.flush()
    elseif line.code == "vocab" then
      io.write("Character '", line.word, "' not in vocabulary", "\n")
      io.flush()
    else
      io.write(line)
      io.flush()
      io.write("Failed, try again\n")
      io.flush()
    end
  else
  	charID = vocab[ line[1] ]
  	local x = torch.Tensor({charID}):expand(params.batch_size):cuda()
  	local y = torch.Tensor{ torch.random(params.vocab_size) }:cuda()
  	local logProbs
  	logProbs, _,  model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
  	g_replace_table(model.s[0], model.s[1])
  	for i = 1, logProbs[1]:size(1) do
  		io.write(logProbs[1][i], ' ')
  	end
  	io.write('\n')
  	io.flush()
  end
end
g_enable_dropout(model.rnns) --]]
