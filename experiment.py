import json

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

from torch.utils.data import DataLoader
from src.data.ring_synthetic import RingCountingDataset
from src.model import FSNetwork
from tqdm import tqdm


# Experimental Values

K_VALUES = [1, 2, 4, 8, 16]
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 15
LR = 1e-3
WIDTH = 0.25

device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f"Device:{device}\n")


# Helpers

train_ds = RingCountingDataset(n_events = 10000, grid_size = 32, max_rings = 3, hits_per_ring = (15, 30), r_range = (0.15, 0.5), smear = 0.01, noise_rate = 0.01, seed = 0)
test_ds = RingCountingDataset(n_events = 10000, grid_size = 32, max_rings = 3, hits_per_ring = (15, 30), r_range = (0.15, 0.5), smear = 0.01, noise_rate = 0.01, seed = 1)

train_dl = DataLoader(train_ds, batch_size = 64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size = 128, shuffle=False)

print(f"Loaded Completed!")
print(f"Train's Lenght: {len(train_ds)}")
print(f"Test's Lenght: {len(test_ds)}\n")

def train_and_eval(K, seed, epochs=EPOCHS, verbose=False):

    torch.manual_seed(seed)
    model = FSNetwork(input=1024, output=3, K=K, width=WIDTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(epochs):
        loss_ = 0
        loop = tqdm(train_dl, desc=f"K={K} seed={seed} ep {epoch+1}/{epochs}", leave=False)
        for x, y in loop:
            optimizer.zero_grad()
            x = x.reshape(x.shape[0], -1).to(device)
            y = y.to(device)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            loss_ += loss.item()
        if verbose:
            print(f"  Loss:{loss_/len(train_dl):.4f}, Epoch: {epoch+1}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_dl:
            x = x.reshape(x.shape[0], -1).to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    return correct / total


# Accuracy vs K 
print("=" * 60)
print(f"Accuracy vs K  --  full grid: {len(K_VALUES)} K x {len(SEEDS)} seeds "
      f"= {len(K_VALUES)*len(SEEDS)} runs")
print("=" * 60)
 
acc = torch.zeros(len(K_VALUES), len(SEEDS))
for i, K in enumerate(K_VALUES):
    for j, seed in enumerate(SEEDS):
        acc[i, j] = train_and_eval(K, seed)
    print(f"K={K:<3d} -> {acc[i].mean()*100:.2f}% +/- {acc[i].std()*100:.2f}%   "
          f"(seeds: {', '.join(f'{a*100:.1f}' for a in acc[i])})")
 
means, stds = acc.mean(dim=1), acc.std(dim=1)
 
best_i = int(means.argmax())
threshold = means[best_i] - stds[best_i]
chosen_i = int((means >= threshold).nonzero()[0])   # K_VALUES is ascending
chosen_K = K_VALUES[chosen_i]
 
print(f"\nBest mean : K={K_VALUES[best_i]} ({means[best_i]*100:.2f}%)")
print(f"Threshold : {threshold*100:.2f}%  (best mean - 1 std)")
print(f"Selected  : K={chosen_K} ({means[chosen_i]*100:.2f}% "
      f"+/- {stds[chosen_i]*100:.2f}%)  -- smallest K within noise of the best")
 
spread = (means.max() - means.min()) * 100
print(f"\nSpread across K: {spread:.2f} pp | mean seed std: {stds.mean()*100:.2f} pp")
if spread < 2 * stds.mean() * 100:
    print("=> Accuracy is flat in K within seed variability.")
else:
    print("=> K has an effect beyond seed variability.")
 
 
# Plots

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
 
ax[0].errorbar(K_VALUES, (means * 100).tolist(), yerr=(stds * 100).tolist(), marker='o', capsize=4)
ax[0].axhline(means[best_i].item() * 100, color='k', ls='--', lw=1, label=f"best mean (K={K_VALUES[best_i]})")

ax[0].axhspan((means[best_i] - stds[best_i]).item() * 100,
              (means[best_i] + stds[best_i]).item() * 100,
              color='k', alpha=0.08, label="best mean +/- 1 std")
              
ax[0].scatter([chosen_K], [means[chosen_i].item() * 100], s=140,
              facecolors='none', edgecolors='tab:red', lw=2,
              label=f"selected K={chosen_K}", zorder=5)
              
ax[0].set_xscale("log", base=2)
ax[0].set_xticks(K_VALUES)
ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0].set_xlabel("K (time steps)")
ax[0].set_ylabel("Test accuracy [%]")
ax[0].set_title(f"Accuracy vs K (mean +/- std over {len(SEEDS)} seeds)")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
 
for j, seed in enumerate(SEEDS):
    ax[1].plot(K_VALUES, (acc[:, j] * 100).tolist(), marker='.', alpha=0.6,
               label=f"seed {seed}")
ax[1].set_xscale("log", base=2)
ax[1].set_xticks(K_VALUES)
ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1].set_xlabel("K (time steps)")
ax[1].set_ylabel("Test accuracy [%]")
ax[1].set_title("Individual seeds (curves cross -> no consistent trend)")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
 
fig.tight_layout()
fig.savefig("figures/accuracy_vs_K.png", dpi=150, bbox_inches="tight")
plt.close(fig)
 
results = {
    "K_values": K_VALUES,
    "seeds": SEEDS,
    "accuracy_matrix": acc.tolist(),
    "mean": means.tolist(),
    "std": stds.tolist(),
    "selected_K": chosen_K,
}
with open("results_K_sweep.json", "w") as f:
    json.dump(results, f, indent=2)
 
print("\nSaved: figures/accuracy_vs_K.png, results_K_sweep.json")

