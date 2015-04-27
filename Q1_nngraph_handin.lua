-- The equation we implement is z = x1 + x2  linear(x3)
-- we have to make an assumption on the size of the inputs
require 'nngraph';
inputSize = 10

x1 = nn.Identity()()
x2 = nn.Identity()()
x3 = nn.Identity()()
linNode = nn.Linear(inputSize, inputSize)(x3)

mulNode = nn.CMulTable()({x2, linNode})
addNode = nn.CAddTable()({x1, mulNode})

m = nn.gModule({x1, x2,x3}, {addNode})


i1 = torch.range(1, 10)
i2 = torch.randn(10)
i3 = torch.randn(10)

print(m:forward({i1, i2, i3}))
