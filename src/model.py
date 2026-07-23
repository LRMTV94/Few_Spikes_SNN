import torch
import torch.nn as nn

torch.manual_seed(42)
device='cuda' if torch.cuda.is_available() else "cpu"
print(device)

class TriangularSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, width):
        ctx.save_for_backward(x)
        ctx.width = width
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        w = ctx.width
        surrogate = torch.clamp(1.0 - x.abs() / w, min=0.0)
        return grad_out * surrogate, None

spike = TriangularSpike.apply


class FSNeuron(nn.Module):
  def __init__(self, K, surrogate_width):
    super().__init__()
    self.K = K
    self.width = surrogate_width
    g = 2.0 ** -(torch.arange(K, dtype=torch.float32))
    self.register_buffer("T", g.clone())
    self.register_buffer("d", g.clone())
    self.register_buffer("h", g.clone())

  def forward(self, x):
    v = x
    out = torch.zeros_like(x)
    spike_count = torch.zeros_like(x)

    for i in range(self.K):
      s = spike(v - self.T[i], self.width)
      out = out + s * self.d[i]
      v = v - s * self.h[i]
      
      spike_count = spike_count + s 
    self.last_spike_count = spike_count.detach()
    
    return out

class FSNetwork(nn.Module):
  def __init__(self, input, output, K, width):
    super().__init__()
    self.K = K
    self.net = nn.Sequential(
       nn.Linear(input,256),
       nn.BatchNorm1d(256),
       FSNeuron(K=K,surrogate_width = width),
       nn.Linear(256,128),
       nn.BatchNorm1d(128),
       FSNeuron(K=K,surrogate_width = width),
       nn.Linear(128,64),
       nn.BatchNorm1d(64),
       FSNeuron(K=K,surrogate_width = width),
       nn.Linear(64,output),
    )

  def forward(self,x):
    return self.net(x)

if __name__ == "__main__":

    x = torch.randn(8, 1, 32, 32)
    net = FSNetwork(input = 1024, output = 3, K = 4, width = 0.25).to(device)
    x = x.reshape(x.shape[0],-1).to(device)
    
    print(net(x).shape)        		  # torch.Size([8, 3])
    print(f"Total parameters: {sum(p.numel() for p in net.parameters()):,}")
