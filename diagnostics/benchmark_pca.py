from matplotlib import colormaps as cm
import torch
import numpy as np
from repsim import AngularShapeMetric
import pandas as pd
from pathlib import Path
import argparse
import time
from sklearn.decomposition import PCA
from tqdm.auto import trange, tqdm
import warnings


parser = argparse.ArgumentParser()
parser.add_argument("--m", type=int, default=1000)
parser.add_argument("--max-p", type=int, default=3000)
parser.add_argument("--num-p", type=int, default=15)
parser.add_argument("--runs", type=int, default=4)
parser.add_argument("--device", default="cpu")
parser.add_argument("--hiddens-file", type=Path, default=None)
parser.add_argument("--seed", type=int, default=24367813)
parser.add_argument(
    "--save-dir", type=Path, default=Path(__file__).parent / "benchmark_pca"
)
parser.add_argument("--cmap", default="tab10")
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()


pvals = np.logspace(np.log10(10), np.log10(args.max_p), args.num_p).round().astype(int)
pvals = pvals[pvals <= args.m]
cmap = cm.get_cmap(args.cmap)


torch.manual_seed(args.seed)
if args.hiddens_file is None:
    suffix = "_random"
    dtype = torch.float32
    h, w, f = 16, 16, 32
    hiddens = {
        "random1": torch.randn(args.m, h, w, f, device=args.device, dtype=dtype),
        "random2": torch.randn(
            args.m, h // 2, w // 2, f * 2, device=args.device, dtype=dtype
        ),
        "random3": torch.randn(
            args.m, h // 4, w // 4, f * 3, device=args.device, dtype=dtype
        ),
        "random4": torch.randn(args.m, 10, device=args.device, dtype=dtype),
    }
else:
    suffix = "_real_data"
    hiddens = torch.load(args.hiddens_file, map_location=args.device)
    hiddens = {k: v[: args.m] for k, v in hiddens.items()}

args.save_dir.mkdir(exist_ok=True)
save_file = args.save_dir / f"results_{args.device}{suffix}.pt"


SOLVERS = ["full", "full-some", "arpack", "randomized"]


def dim_reduce(x, p, method="full"):
    x = x.reshape(x.shape[0], -1)
    x = x - x.mean(dim=0, keepdim=True)
    if method == "full":
        _, _, vT = torch.linalg.svd(x)
        return torch.einsum("mn,pn->mp", x, vT[:p, :])
    elif method == "full-some":
        _, _, vT = torch.linalg.svd(x, full_matrices=False)
        return torch.einsum("mn,pn->mp", x, vT[:p, :])
    elif method == "arpack":
        return torch.tensor(
            PCA(n_components=p, svd_solver="arpack")
            .fit_transform(x.cpu())
            .astype(np.float32),
            device=args.device,
        )
    elif method == "randomized":
        return torch.tensor(
            PCA(n_components=p, svd_solver="randomized")
            .fit_transform(x.cpu())
            .astype(np.float32),
            device=args.device,
        )


# %%
results = []
if save_file.exists():
    precomputed = torch.load(save_file)
else:
    precomputed = {}

for p in tqdm(pvals, desc="p", position=0, leave=False):
    metric = AngularShapeMetric(m=args.m, p=p, alpha=1.0)
    for name, hidden in hiddens.items():
        for r in trange(args.runs, desc=name, position=1, leave=False):
            reference_x = None
            for method in SOLVERS:
                save_key = f"{name}_{method}_{p}_{r}"
                if save_key in precomputed:
                    telapsed, length = precomputed[save_key]
                else:
                    try:
                        with torch.no_grad():
                            tstart = time.time()
                            x = dim_reduce(hidden, p, method=method)
                            reference_x = x if reference_x is None else reference_x
                            telapsed = time.time() - tstart
                        with warnings.catch_warnings():
                            length = metric.length(
                                reference_x / torch.linalg.norm(reference_x, ord="fro"),
                                x / torch.linalg.norm(x, ord="fro"),
                            )
                    except (RuntimeError, ValueError) as e:
                        print(
                            f"==============================\nError ({p}, {name}, {r}, {method}):\n{e}"
                        )
                        telapsed, length = np.nan, np.nan
                    precomputed[save_key] = (telapsed, length)
                    torch.save(precomputed, save_file)

                results.append(
                    {
                        "method": method,
                        "layer": name,
                        "p": p,
                        "time": telapsed,
                        "dist": length.item() if torch.is_tensor(length) else length,
                        "run": r,
                    }
                )

results = pd.DataFrame(results)

if args.plot:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 4))
    sns.lineplot(
        data=results,
        x="p",
        y="time",
        hue="method",
        style="layer",
        markers=True,
        dashes=False,
    )
    plt.title(f"Time to compute PCA on {args.device} for various dim reduction methods")
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.lineplot(
        data=results,
        x="p",
        y="dist",
        hue="method",
        style="layer",
        markers=True,
        dashes=False,
    )
    plt.title(
        f"ShapeMetric.length(x, method(x)) on {args.device} for various dim reduction methods"
    )
    plt.show()
