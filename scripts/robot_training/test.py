import torch
import torch.nn.functional as F
import numpy as np


class DynamicNet(torch.nn.Module):
  def __init__(self, D_in, D_H, N_H, D_out):

    super(DynamicNet, self).__init__()
    self.input_linear = torch.nn.Linear(D_in, D_H)
    self.middle_linear = torch.nn.Linear(D_H, D_H)
    self.output_linear = torch.nn.Linear(D_H, D_out)
 
  def forward(self, x):

    #h_relu = self.input_linear(x).clamp(min=0)
    #for _ in range(N_H):
    #  h_relu = self.middle_linear(h_relu).clamp(min=0)
    #y_pred = self.output_linear(h_relu)

    h_relu = F.tanh(self.input_linear(x))
    for _ in range(N_H):
        h_relu = F.tanh(self.middle_linear(h_relu))
    y_pred = F.tanh(self.output_linear(h_relu))

    return y_pred


# N is batch size; D_in is input dimension;
# D_H is hidden dimension; D_out is output dimension.
N, D_in, D_H, N_H, D_out = 5, 5, 10, 3, 3

# Create random Tensors to hold inputs and outputs.
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, D_H, N_H, D_out)
if torch.cuda.is_available():
  device = 'cuda'
  if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
else:
  device = 'cpu'

#model = torch.nn.DataParallel(model).cuda()
model = model.to(device)
# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(size_average=False)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

t = 0
#for t in range(500):
while True:

  # Forward pass: Compute predicted y by passing x to the model
  x = x.to(device)
  y_pred = model(x)

  # Compute and print loss
  loss = criterion(y_pred.cpu(), y)
  print(t)
  print(np.round(y.data.cpu().numpy()-y_pred.data.cpu().numpy(), 3))
  print(loss.item())
  print("-------------------------------------------")

  # Zero gradients, perform a backward pass, and update the weights.
  optimizer.zero_grad()  # zero the gradient buffers
  loss.backward()
  optimizer.step()  # Does the update
  t += 1


