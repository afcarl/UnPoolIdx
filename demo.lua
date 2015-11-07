require 'nn'
require 'cunn'
require 'UnPoolIdx'
local function main()

  -- set up net
  -- need to extract pool indices from the corresponding pooling layer
  local net = nn.Sequential()
  local p = nn.SpatialMaxPooling(2,2)
  local idx = p.indices
  net:add(p)
  net:add(nn.UnPoolIdx(2,idx))

  -- set up input
  input = torch.Tensor(3,4,6)
  for i=1,3 do
    for j=1,4 do
      for k=1,6 do
        input[i][j][k] = j*k
      end
    end
  end
  
  -- print stuff and run forward and backward
  print 'Input'
  print (input)
  print 'After pooling'
  print (p:forward(input))
  o = net:forward(input)
  print 'Output'
  print (o)
  print 'Idx'
  print (idx)
  gradOutput = torch.Tensor(3,4,6)
  for i=1,3 do
    for j=1,4 do
      for k=1,6 do
        gradOutput[i][j][k] = j*k
      end
    end
  end
  print 'gradOutput'
  print (gradOutput)
  gi = net:updateGradInput(input, gradOutput)
  print 'gradInput'
  print (gi)
end

main()
