import torch
import numpy as np
from tests.metrics import _list_of_metrics
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm


m, n, runs = 10000, 300, 10
mvals = np.logspace(2, np.log10(m), 31).round().astype(int)
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

data_x = torch.randn(m, n, device=device, dtype=dtype)
data_y = data_x + 2 * torch.randn(m, n, device=device, dtype=dtype) / np.sqrt(n)

#%%

save_dir = Path(__file__).parent / "bias_variance"
save_dir.mkdir(exist_ok=True)

#%%
metrics = [m["metric"] for m in _list_of_metrics]
for i, metric in enumerate(metrics):
    if i == 0: continue
    print("Starting", metric.string_id())
    lengths = torch.zeros(len(mvals), runs)
    for i, sub_m in tqdm(enumerate(mvals), desc=metric.string_id(), total=len(mvals)):
        metric.m = sub_m
        for j in range(runs):
            idx = torch.randperm(m)[:sub_m]
            sub_x, sub_y = data_x[idx, :], data_y[idx, :]
            try:
                lengths[i, j] = metric.length(*map(metric.neural_data_to_point, [sub_x, sub_y]))
            except:
                lengths[i, j] = np.nan


    lengths = lengths.detach().cpu().numpy()
    plt.figure()
    mu, sigma = np.nanmean(lengths, axis=-1), np.nanstd(lengths, axis=-1)
    plt.fill_between(mvals, mu-3*sigma, mu+3*sigma, color=(0., 0., 0., 0.25))
    plt.plot(mvals, mu, color=(0., 0., 0.))
    plt.xscale('log')
    plt.xlabel('m')
    plt.ylabel('length')
    plt.title(metric.string_id())
    plt.savefig(save_dir / (metric.string_id() + ".svg"))
    plt.show()
