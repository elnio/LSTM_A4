stringx = require('pl.stringx')
require 'io'
require 'cunn'
cutorch.setDevice(1)
require 'nngraph'
require('base')

params = {batch_size=20,
		  seq_length=20,
		  layers=2,
		  rnn_size=200,
		  vocab_size=10000}
vocabName = 'vocabWord.t7b'
vocab = torch.load(vocabName)
inverseVocabName = 'inverseVocabWord.t7b'
invVocab = torch.load(inverseVocabName)

coreNetName = 'wordCoreNet.t7b'
coreNet = torch.load(coreNetName)

function transfer_data(x)
  return x:cuda()
end

function setup()
  local core_network = coreNet
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
  if tonumber(line[1]) == nil then error({code="init"}) end
  for i = 2,#line do
    if vocab[line[i]] == nil then error({code="vocab", word = line[i]}) end
  end
  return line
end

model = {}
setup()
while true do
  io.write("Query: len word1 word2 etc\n")
  io.flush()
  local ok, line = pcall(readline)
  if not ok then
    if line.code == "EOF" then
      break -- end loop
    elseif line.code == "vocab" then
      io.write("Word not in vocabulary ", line.word, "\n")
      io.flush()
    elseif line.code == "init" then
      io.write("Start with a number\n")
      io.flush()
    else
      io.write(line)
      io.flush()
      io.write("Failed, try again\n")
      io.flush()
    end
  else
  	-- set up the model for testing
  	g_disable_dropout(model.rnns)
  	if model ~= nil and model.start_s ~= nil then
	  	for d = 1, 2 * params.layers do
    	  model.start_s[d]:zero()
    	end
    end
    g_replace_table(model.s[0], model.start_s)
  	
  	local idPredWord
  	-- The first loop, over the input words, simply changes the state of the model.
  	for i =2, #line do
  		local x = vocab[line[i]]
  		x = torch.Tensor({x}):expand(params.batch_size):cuda()
  		local y = torch.Tensor{ torch.random(params.vocab_size) }:cuda()
  		local logProbs
  		logProbs, _, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
  		idPredWord = torch.multinomial(torch.exp(logProbs[1]:double()), 1)[1]
  		g_replace_table(model.s[0], model.s[1])
  		io.write(line[i], ' ')
  	end
  	
  	-- The second loop, predicts the words.
  	for i =1, line[1] do
  		-- first we write to the output the prediction we made before
  		io.write(invVocab[idPredWord], ' ')
  		-- and then we use it to form the new prediction.
  		local x = torch.Tensor({idPredWord}):expand(params.batch_size):cuda()
  		local y = torch.Tensor{ torch.random(params.vocab_size) }:cuda()
  		local logProbs
  		logProbs, _, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
  		idPredWord = torch.multinomial(torch.exp(logProbs[1]:double()), 1)[1]
  		g_replace_table(model.s[0], model.s[1])

  	end  	
	g_enable_dropout(model.rnns)
  	io.write('\n')
  	io.flush()
  end
end
