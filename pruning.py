import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.ring_synthetic import RingCountingDataset
from src.model import FSNetwork, FSNeuron
from torch.utils.data import DataLoader
from tqdm import tqdm


# Experimental Parameters
K = 4
WIDTH = 0.25
EPOCHS = 15
FINETUNE_EPOCHS = 5
LR = 1e-3
SEEDS = [0, 1, 2]
RATIOS = [0.0, 0.5, 0.7, 0.8, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98]
MODE = "layer"

device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")


# Loading Syntethic Dataset

train_ds = RingCountingDataset(n_events=10000, grid_size=32, max_rings=3, hits_per_ring=(15, 30), r_range=(0.15, 0.5), smear=0.01, noise_rate=0.01, seed=0)
test_ds = RingCountingDataset(n_events=10000, grid_size=32, max_rings=3, hits_per_ring=(15, 30), r_range=(0.15, 0.5), smear=0.01, noise_rate=0.01, seed=1)

print(f"Loaded Completed!")
print(f"Train's Lenght: {len(train_ds)}")
print(f"Test's Lenght:  {len(test_ds)}\n")

train_dl = DataLoader(train_ds, batch_size = 64, shuffle = True)
test_dl = DataLoader(test_ds, batch_size = 128, shuffle = False)


# Helpers

def evaluate(model):
    model.eval()
    fs_layers = [m for m in model.net if isinstance(m, FSNeuron)]
    correct = total = n_batches = 0
    spike_sum = torch.zeros(len(fs_layers))
    silent_sum = torch.zeros(len(fs_layers))

    with torch.no_grad():
        for x, y in test_dl:
            x = x.reshape(x.shape[0], -1).to(device)
            y = y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.shape[0]
            for i, m in enumerate(fs_layers):
                spike_sum[i] += m.last_spike_count.mean().item()
                silent_sum[i] += (m.last_spike_count == 0).float().mean().item()
            n_batches += 1

    return correct / total, (spike_sum / n_batches).mean().item(), \
           (silent_sum / n_batches).mean().item()


def train(model, epochs, lr, desc="train"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        for x, y in tqdm(train_dl, desc=f"{desc} {ep+1}/{epochs}", leave=False):
            opt.zero_grad()
            x = x.reshape(x.shape[0], -1).to(device)
            y = y.to(device)
            F.cross_entropy(model(x), y).backward()
            opt.step()
    return model


def weight_sparsity(model):
    z = n = 0
    for m in model.net:
        if isinstance(m, nn.Linear):
            z += (m.weight == 0).sum().item()
            n += m.weight.numel()
    return z / n


def apply_pruning(model, ratio, mode=MODE):
    """NON chiama prune.remove: le maschere restano attive nel fine-tuning."""
    if ratio == 0.0:
        return model
    layers = [m for m in model.net if isinstance(m, nn.Linear)]
    if mode == "layer":
        for m in layers:
            prune.l1_unstructured(m, name="weight", amount=ratio)
    else:
        prune.global_unstructured([(m, "weight") for m in layers],
                                  pruning_method=prune.L1Unstructured,
                                  amount=ratio)
    return model


def make_permanent(model):
    """Fonde le maschere nei pesi. SOLO dopo il fine-tuning."""
    for m in model.net:
        if isinstance(m, nn.Linear) and hasattr(m, "weight_mask"):
            prune.remove(m, "weight")
    return model


# Sweep: RATIOS x SEEDS

nR, nS = len(RATIOS), len(SEEDS)
acc_one = torch.zeros(nR, nS)      # accuracy one-shot
acc_ft  = torch.zeros(nR, nS)      # accuracy dopo fine-tuning
spk_one = torch.zeros(nR, nS)      # spike/neurone one-shot
spk_ft  = torch.zeros(nR, nS)      # spike/neurone dopo fine-tuning
sil_ft  = torch.zeros(nR, nS)      # frazione di neuroni silenti
spars   = torch.zeros(nR, nS)      # sparsita' effettiva (controllo)
dense   = torch.zeros(nS)          # accuracy della rete densa per seed

print("=" * 68)
print(f"Pruning sweep: {nR} ratios x {nS} seeds = {nR*nS} runs ({MODE}-wise)")
print("=" * 68)

for j, seed in enumerate(SEEDS):
    print(f"\n--- seed {seed}: training dense baseline ---")
    torch.manual_seed(seed)
    base = FSNetwork(input=1024, output=3, K=K, width=WIDTH).to(device)
    train(base, EPOCHS, LR, desc=f"dense s{seed}")
    a0, s0, si0 = evaluate(base)
    dense[j] = a0
    print(f"dense: acc {a0*100:.2f}% | {s0:.2f}/{K} spikes/neuron | "
          f"silent {si0*100:.1f}%")

    for i, ratio in enumerate(RATIOS):
        m = apply_pruning(copy.deepcopy(base), ratio)

        a1, s1, _ = evaluate(m)                        # one-shot
        train(m, FINETUNE_EPOCHS, LR / 10, desc=f"ft s{seed} r{ratio}")
        make_permanent(m)                              # solo ora
        a2, s2, si2 = evaluate(m)
        sp = weight_sparsity(m)                        # controllo

        acc_one[i, j], acc_ft[i, j] = a1, a2
        spk_one[i, j], spk_ft[i, j] = s1, s2
        sil_ft[i, j], spars[i, j] = si2, sp

        flag = "" if abs(sp - ratio) < 0.02 else "  <-- SPARSITA' NON RISPETTATA"
        print(f"  ratio {ratio:4.2f} (sp {sp*100:5.1f}%) | one-shot {a1*100:5.2f}% "
              f"-> fine-tuned {a2*100:5.2f}% | spikes {s1:.2f} -> {s2:.2f}{flag}")


# Riepilogo: media +/- std sui seed

print("\n" + "=" * 68)
print(f"Summary (mean +/- std over {nS} seeds)")
print("=" * 68)
print(f"Dense baseline: {dense.mean()*100:.2f}% +/- {dense.std()*100:.2f}%\n")
print(f"{'ratio':>6} | {'one-shot':>16} | {'fine-tuned':>16} | {'spikes (ft)':>13} | {'silent':>7}")
print("-" * 68)
for i, ratio in enumerate(RATIOS):
    print(f"{ratio:6.2f} | "
          f"{acc_one[i].mean()*100:6.2f}% +/- {acc_one[i].std()*100:4.2f} | "
          f"{acc_ft[i].mean()*100:6.2f}% +/- {acc_ft[i].std()*100:4.2f} | "
          f"{spk_ft[i].mean():5.2f} +/- {spk_ft[i].std():4.2f} | "
          f"{sil_ft[i].mean()*100:5.1f}%")

# soglia: massimo ratio la cui accuracy media resta entro 1 pp dal controllo (ratio 0)
ref = acc_ft[0].mean()
ok = [RATIOS[i] for i in range(nR) if acc_ft[i].mean() >= ref - 0.01]
print(f"\nMax sparsity within 1 pp of the fine-tuned dense control "
      f"({ref*100:.2f}%): {max(ok)*100:.0f}%")


# Plot

r = [x * 100 for x in RATIOS]
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.errorbar(r, (acc_one.mean(1)*100).tolist(), yerr=(acc_one.std(1)*100).tolist(),
            marker='o', ls='--', color="tab:blue", alpha=0.55, capsize=3,
            label="one-shot")
ax.errorbar(r, (acc_ft.mean(1)*100).tolist(), yerr=(acc_ft.std(1)*100).tolist(),
            marker='o', color="tab:blue", capsize=3, label="fine-tuned")
ax.axhline(ref.item()*100, color='k', ls=':', lw=1, label="dense control")
ax.set_xlabel("weight sparsity [%]")
ax.set_ylabel("test accuracy [%]", color="tab:blue")
ax.tick_params(axis='y', labelcolor="tab:blue")
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=8)

ax2 = ax.twinx()
ax2.errorbar(r, spk_ft.mean(1).tolist(), yerr=spk_ft.std(1).tolist(),
             marker='s', color="tab:red", capsize=3, label="spikes/neuron")
ax2.set_ylabel(f"spikes per neuron (of K={K})", color="tab:red")
ax2.tick_params(axis='y', labelcolor="tab:red")
ax2.set_ylim(0, K)
ax2.legend(loc="upper right", fontsize=8)

ax.set_title(f"Accuracy and spike activity vs weight sparsity\n"
             f"({MODE}-wise pruning, mean +/- std over {nS} seeds)")
fig.tight_layout()
fig.savefig("figures/pruning.png", dpi=150, bbox_inches="tight")
plt.close(fig)

with open("results_pruning.json", "w") as f:
    json.dump({"ratios": RATIOS, "seeds": SEEDS, "mode": MODE,
               "dense": dense.tolist(),
               "acc_oneshot": acc_one.tolist(), "acc_finetuned": acc_ft.tolist(),
               "spikes_oneshot": spk_one.tolist(), "spikes_finetuned": spk_ft.tolist(),
               "silent_finetuned": sil_ft.tolist(), "sparsity": spars.tolist()},
              f, indent=2)

print("\nSaved: figures/pruning.png, results_pruning.json")

