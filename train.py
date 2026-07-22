import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sklearn

from torchinfo import summary
from src.data.ring_synthetic import RingCountingDataset
from src.model import FSNetwork
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm.auto import tqdm

torch.manual_seed(42)
device='cuda' if torch.cuda.is_available() else "cpu"
print(f"Device:{device}")

# Loading Syntethic Dataset

train_ds = RingCountingDataset(n_events=5000, grid_size=32, max_rings=3, hits_per_ring=(15, 30), r_range=(0.15, 0.5), smear=0.01, noise_rate=0.01, seed=0)
test_ds = RingCountingDataset(n_events=500, grid_size=32, max_rings=3, hits_per_ring=(15, 30), r_range=(0.15, 0.5), smear=0.01, noise_rate=0.01, seed=0)

print("\n")
print(f"Loaded Completed!")
print(f"Train's Lenght: {len(train_ds)}\n")
print("\n")

train_dl = DataLoader(train_ds, batch_size = 64, shuffle = True)
test_dl = DataLoader(test_ds, batch_size = 128, shuffle = False)

# Training Model

epochs = 15
model = FSNetwork(input=1024, output=3, K=4, width=0.25).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

correct = 0
total = 0
cm = torch.zeros(3, 3, dtype=torch.int64).to(device)
losses=[]

print(summary(model, input_size=(3, 1024)),"\n")
print(f"Parameters:\n")
for name, p in model.named_parameters():
  print(f" Name:{name},\t Parameters:{p.requires_grad}")
print("\n")

print(f"Training+Test MLP-Augmented\n")

for epoch in range(epochs):
  loss_=0
  for x,y in tqdm(train_dl,desc=f"Epoca {epoch+1}/{epochs}"):
    optimizer.zero_grad()
    x = x.reshape(x.shape[0],-1).to(device)
    y = y.to(device)
    pred = model(x)
    loss = F.cross_entropy(pred,y)
    loss_+= loss.item()

    loss.backward()
    optimizer.step()

  losses.append(loss_/len(train_dl))
  print(f"Loss:{(loss_/len(train_dl)):.2f}, Epoch: {epoch+1}")

model.eval()
with torch.no_grad():
    for x,y in test_dl:
      x = x.reshape(x.shape[0], -1).to(device)
      y = y.to(device)
      pred = model(x).argmax(dim=1)
      correct += (pred==y).sum().item()
      total += y.shape[0]

      cm += torch.bincount(y * 3 + pred, minlength=9).reshape(3, 3)

print("\n")
print(f"Total accuracy FS-MLP:{(correct/total)*100:.2f}%\n\n\n")
print("\n")

disp = ConfusionMatrixDisplay(confusion_matrix=cm.cpu().numpy())
disp.plot(cmap='Blues')
disp.ax_.set_ylabel("True Class")
disp.ax_.set_xlabel("Predicted Class")
disp.figure_.savefig("figures/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close(disp.figure_)
print("\n")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(losses, marker='o')
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
fig.savefig("figures/training_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n")
