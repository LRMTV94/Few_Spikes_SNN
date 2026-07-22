import torch
import math
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

def octagon_mask(n):
    cut = n // 4
    m = torch.ones(n, n, dtype=torch.bool)
    for i in range(cut):
        m[i, :cut - i] = m[i, n - cut + i:] = False
        m[n - 1 - i, :cut - i] = m[n - 1 - i, n - cut + i:] = False
    return m

class RingCountingDataset(Dataset):
    def __init__(self, n_events, grid_size, max_rings, hits_per_ring, r_range, smear, noise_rate, seed):

        g = torch.Generator().manual_seed(seed)
        self.mask = octagon_mask(grid_size)

        X = torch.zeros(n_events, grid_size, grid_size)
        y = torch.randint(1, max_rings + 1, (n_events,), generator=g)

        for i in range(n_events):
            for _ in range(int(y[i])):

                cx, cy = 0.25 + 0.5 * torch.rand(2, generator=g)

                r = r_range[0] + (r_range[1] - r_range[0]) * torch.rand(1, generator=g)
                nh = int(torch.randint(hits_per_ring[0], hits_per_ring[1], (1,), generator=g))
                th = 2 * math.pi * torch.rand(nh, generator=g)

                px = cx + r * torch.cos(th) + smear * torch.randn(nh, generator=g)
                py = cy + r * torch.sin(th) + smear * torch.randn(nh, generator=g)

                ix = (px * grid_size).long().clamp(0, grid_size - 1)
                iy = (py * grid_size).long().clamp(0, grid_size - 1)
                X[i, iy, ix] = 1.0

            noise = (torch.rand(grid_size, grid_size, generator=g) < noise_rate)
            X[i] = (X[i] + noise).clamp(0, 1) * self.mask

        self.X = X.unsqueeze(1)
        self.y = (y - 1).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
if __name__ == "__main__":

    ds = RingCountingDataset(n_events=2000, grid_size=32, max_rings=3, hits_per_ring=(15, 30), r_range=(0.15, 0.5), smear=0.01, noise_rate=0.01, seed=42)
    print(f"{len(ds)} events | shape {tuple(ds[0][0].shape)}")
    print(f"Label distribution: {torch.bincount(ds.y).tolist()}")

    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(6, 6))
    axes = axes.flatten()

    for ax, i in zip(axes.flatten(), range(8)):
      img, lbl = ds[i]
      axes[i].imshow(img.squeeze(), cmap='gray', interpolation='nearest')
      axes[i].set_title(f"Label: {int(lbl)}", fontsize=10)
      axes[i].axis('off')

    plt.tight_layout()
    plt.show()


