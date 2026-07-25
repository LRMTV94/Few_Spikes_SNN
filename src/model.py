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

class Network(nn.Module):
  def __init__(self, input_, output):
    super().__init__()
    self.net = nn.Sequential(
    
       nn.Linear(input_, 256),
       nn.BatchNorm1d(256),
       nn.ReLU(),
       
       nn.Linear(256, 128),
       nn.BatchNorm1d(128),
       nn.ReLU(),
       
       nn.Linear(128, 64),
       nn.BatchNorm1d(64),
       nn.ReLU(),
       
       nn.Linear(64,output),
    )

  def forward(self, x):
    return self.net(x.flatten(1))


class FSNetwork(nn.Module):
  def __init__(self, input_, output, K, width):
    super().__init__()   
    self.K = K
    
    self.net = nn.Sequential(
       nn.Linear(input_, 256),
       nn.BatchNorm1d(256),
       FSNeuron(K=K,surrogate_width = width),
       
       nn.Linear(256, 128),
       nn.BatchNorm1d(128),
       FSNeuron(K=K,surrogate_width = width),
       
       nn.Linear(128, 64),
       nn.BatchNorm1d(64),
       FSNeuron(K=K,surrogate_width = width),
       
       nn.Linear(64, output),
    )

  def forward(self, x):
    return self.net(x.flatten(1))

class FSConvNetwork(nn.Module):
    def __init__(self, input_ch, output, K, width):
        super().__init__()
        self.K = K
        self.features = nn.Sequential(
            nn.Conv2d(input_ch, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            FSNeuron(K, width),
            nn.MaxPool2d(2),                     # 32x32 -> 16x16

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            FSNeuron(K, width),
            nn.MaxPool2d(2),                     # 16x16 -> 8x8
        )
        self.classifier = nn.Linear(32 * 8 * 8, output)

    def forward(self, x):
        x = self.features(x)                     
        return self.classifier(x.flatten(1))
        
if __name__ == "__main__":

    x = torch.randn(8, 1, 32, 32)
    net_mlp = Network(input_ = 1024, output = 3).to(device)
    x = x.to(device)
    
    print(net_mlp(x).shape)        		  # torch.Size([8, 3])
    print(f"Total MLP parameters: {sum(p.numel() for p in net_mlp.parameters()):,}\n")

    x = torch.randn(8, 1, 32, 32)
    net = FSNetwork(input_ = 1024, output = 3, K = 4, width = 0.25).to(device)
    x = x.to(device)
    
    print(net(x).shape)        		  # torch.Size([8, 3])
    print(f"Total FS-MLP parameters: {sum(p.numel() for p in net.parameters()):,}\n")
    
    x = torch.randn(8, 1, 32, 32)
    net_cnn = FSConvNetwork(input_ch = 1, output = 3, K = 4, width = 0.25).to(device)
    
    print(net_cnn(x).shape)        		  # torch.Size([8, 3])
    print(f"Total FS-CNN parameters: {sum(p.numel() for p in net_cnn.parameters()):,}")
