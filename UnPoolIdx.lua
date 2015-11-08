local UnPoolIdx, parent = torch.class('nn.UnPoolIdx', 'nn.Module')

function UnPoolIdx:__init(s,indices)
   parent.__init(self)

   self.scale = s

   self.indices = indices
end

function UnPoolIdx:updateOutput(input)
   assert (input:nDimension() == 3 or input:nDimension() == 4)
   assert (self.indices:nDimension() == input:nDimension())
   assert (self.indices:nElement() == input:nElement()) 
   --input = input:float()
   local bsize, H, W, oH, oW, D
   if input:nDimension() == 4 then
     bsize,D,H,W = input:size(1),input:size(2),input:size(3),input:size(4)
   else
     D,H,W = input:size(1),input:size(2),input:size(3)
   end

   local oH = H*self.scale
   local oW = W*self.scale
   if bsize then
     self.output:resize(bsize, D, oH, oW):zero()
   else
     self.output:resize(D, oH, oW):zero()
   end
   
   local i,j
   if bsize then
     for b = 1,bsize do
       for d=1,D do
         for h=1,H do
           for w=1,W do
             i = math.floor(self.indices[b][d][h][w]/oW)+1 
             j = math.fmod(self.indices[b][d][h][w],oW)
             if j==0 then 
               j = oW 
               i = i-1 
             end
             self.output[b][d][i][j] = input[b][d][h][w]
           end
         end
       end
     end
   else
     for d=1,D do
       for h=1,H do
         for w=1,W do
           i = math.floor(self.indices[d][h][w]/oW)+1
           j = math.fmod(self.indices[d][h][w],oW)
           if j==0 then 
             j = oW 
             i = i-1 
           end
           self.output[d][i][j] = input[d][h][w]
         end
       end
     end
   end

   return self.output
end

function UnPoolIdx:updateGradInput(input, gradOutput)
   local bsize, H, W, oH, oW, D
   if input:nDimension() == 4 then
     bsize,D,H,W = input:size(1),input:size(2),input:size(3),input:size(4)
   else
     D,H,W = input:size(1),input:size(2),input:size(3)
   end

   if bsize then
     self.gradInput:resize(bsize, D, H, W):zero()
   else
     self.gradInput:resize(D, H, W):zero()
   end

   local oH = H*self.scale
   local oW = W*self.scale
   
   local i,j
   if bsize then
     for b = 1,bsize do
       for d=1,D do
         for h=1,H do
           for w=1,W do
             i = math.floor(self.indices[b][d][h][w]/oW)+1
             j = math.fmod(self.indices[b][d][h][w],oW)
             if j==0 then 
               j = oW 
               i = i-1 
             end
             self.gradInput[b][d][h][w] = gradOutput[b][d][i][j]
           end
         end
       end
     end
   else
     for d=1,D do
       for h=1,H do
         for w=1,W do
           i = math.floor(self.indices[d][h][w]/oW)+1
           j = math.fmod(self.indices[d][h][w],oW)
           if j==0 then 
             j = oW 
             i = i-1 
           end
           self.gradInput[d][h][w] = gradOutput[d][i][j]
         end
       end
     end
   end
   
  return self.gradInput
end
