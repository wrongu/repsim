import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import torch
import numpy as np
from repsim.geometry.trig import angle
from repsim import AngularCKA
from repsim.kernels import SquaredExponential
from pathlib import Path
from tqdm.auto import trange

#%%
m, n, runs = 10000, 300, 10
mvals = np.logspace(2, np.log10(m), 31).round().astype(int)
device = "cuda:2" if torch.cuda.is_available() else "cpu"
print("Running on", device)

source_file = Path("tests") / "test_data" / "example_hiddens.pth"
if source_file.exists():
    hiddens = torch.load(source_file, map_location=device)
    data_x = hiddens["blocks.2.relu2.in"]
    data_y = hiddens["blocks.3.relu2.in"]
    data_z = hiddens["blocks.4.relu2.in"]
    assert len(data_x) >= m and len(data_y) >= m
else:
    dtype = torch.float32
    data_x = torch.randn(m, n, device=device, dtype=dtype)
    data_y = data_x + 2 * torch.randn(m, n, device=device, dtype=dtype) / np.sqrt(n)
    data_z = data_x + 2 * torch.randn(m, n, device=device, dtype=dtype) / np.sqrt(n)

#%%

save_dir = Path() / "bias_variance"
save_dir.mkdir(exist_ok=True)

#%%
metrics = [
    AngularCKA(m=100),
    AngularCKA(m=100, kernel=SquaredExponential()),
    AngularCKA(m=100, use_unbiased_hsic=True),
    AngularCKA(m=100, kernel=SquaredExponential(), use_unbiased_hsic=True),
]
cmap = cm.get_cmap("tab10")
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for im, metric in enumerate(metrics):
    savename = save_dir / f"bias_variance_{metric.string_id()}.pth"
    if savename.exists():
        print("Loading", metric.string_id())
        data = torch.load(savename)
        lengths = data["lengths"]
    else:
        print("Starting", metric.string_id())
        lengths = torch.zeros(len(mvals), runs)
        angles = torch.zeros(len(mvals), runs)
        for j in trange(runs, desc=metric.string_id()):
            for i, sub_m in enumerate(mvals):
                metric.m = sub_m
                idx = torch.randperm(m)[:sub_m]
                sub_x, sub_y, sub_z = data_x[idx], data_y[idx], data_z[idx]
                try:
                    lengths[i, j] = metric.length(*map(metric.neural_data_to_point, [sub_x, sub_y]))
                    angles[i, j] = angle(metric, *map(metric.neural_data_to_point, [sub_x, sub_y, sub_z]))
                except:
                    lengths[i, j] = np.nan
                    angles[i, j] = np.nan
        lengths = lengths.detach().cpu().numpy()
        angles = angles.detach().cpu().numpy()
        torch.save({"lengths": lengths, "angles": angles}, savename)

    c = cmap(im/10)
    mu, sigma = np.nanmean(lengths, axis=-1), np.nanstd(lengths, axis=-1)
    axs[0].fill_between(mvals, mu-3*sigma, mu+3*sigma, color=c[:3] + (0.25,))
    axs[0].plot(mvals, mu, color=c, label=".".join(metric.string_id().split(".")[:-1]))
    mu, sigma = np.nanmean(angles, axis=-1), np.nanstd(angles, axis=-1)
    axs[1].fill_between(mvals, mu-3*sigma, mu+3*sigma, color=c[:3] + (0.25,))
    axs[1].plot(mvals, mu, color=c, label=".".join(metric.string_id().split(".")[:-1]))
axs[0].set_xscale("log")
axs[0].set_xlabel("m")
axs[0].set_ylabel("length(x,y)")
axs[1].set_xscale("log")
axs[1].set_xlabel("m")
axs[1].set_ylabel("angle(x,y,z)")
axs[0].legend()
fig.tight_layout()
plt.show()
