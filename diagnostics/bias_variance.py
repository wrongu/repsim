import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import torch
import numpy as np
from repsim.geometry.trig import angle
from repsim import AngularCKA, AngularShapeMetric, EuclideanShapeMetric
from pathlib import Path
from tqdm.auto import trange
from collections import defaultdict
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--p", type=int, default=10)
parser.add_argument("--max-m", type=int, default=10000)
parser.add_argument("--num-m", type=int, default=15)
parser.add_argument("--runs", type=int, default=10)
parser.add_argument("--device", default="cpu")
parser.add_argument("--hiddens-file", type=Path, default=None)
parser.add_argument("--seed", type=int, default=24367813)
parser.add_argument("--save-dir", type=Path, default=Path(__file__).parent / "bias_variance")
parser.add_argument("--cmap", default="tab10")
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()


mvals = np.logspace(np.log10(100), np.log10(args.max_m), args.num_m).round().astype(int)
cmap = cm.get_cmap(args.cmap)


torch.manual_seed(args.seed)
if args.hiddens_file is None:
    suffix = "_random"
    dtype = torch.float32
    h, w, f = 16, 16, 32
    data_x = torch.randn(args.max_m, h, w, f, device=args.device, dtype=dtype)
    data_y = data_x + 2 * torch.randn(args.max_m, h, w, f, device=args.device, dtype=dtype) / np.sqrt(f)
    data_z = data_x + 2 * torch.randn(args.max_m, h, w, f, device=args.device, dtype=dtype) / np.sqrt(f)
    data_l = torch.randn(args.max_m, 10, device=args.device, dtype=dtype)
else:
    suffix = "_real_data"
    hiddens = torch.load(args.hiddens_file, map_location=args.device)
    # x y and z are all conv layers
    data_x = hiddens["blocks.2.relu2.in"]
    data_y = hiddens["blocks.3.relu2.in"]
    data_z = hiddens["blocks.4.relu2.in"]
    # l is logits
    try:
        data_l = hiddens["logits"]
    except KeyError:
        data_l = torch.randn(data_x.shape[0], 10, device=args.device)
    assert len(data_x) >= args.max_m and len(data_y) >= args.max_m and len(data_z) >= args.max_m and len(data_l) >= args.max_m

args.save_dir.mkdir(exist_ok=True)

#%%
sets_of_metrics = [
    [
        AngularCKA(m=100),
        AngularCKA(m=100, use_unbiased_hsic=False),
    ],
    [
        AngularShapeMetric(m=100, p=10, alpha=1.0),
        AngularShapeMetric(m=100, p=100, alpha=1.0),
        AngularShapeMetric(m=100, p=1000, alpha=1.0)
        # AngularShapeMetric(m=100, p=None, alpha=1.0)  # TODO
    ],
    # [
    #     AngularShapeMetric(m=100, p=10, alpha=0.0),
    #     AngularShapeMetric(m=100, p=100, alpha=0.0),
    #     AngularShapeMetric(m=100, p=1000, alpha=0.0),
    #     # AngularShapeMetric(m=100, p=None, alpha=0.0),  # TODO
    # ],
    # [
    #     EuclideanShapeMetric(m=100, p=10, alpha=1.0),
    #     EuclideanShapeMetric(m=100, p=100, alpha=1.0),
    #     EuclideanShapeMetric(m=100, p=1000, alpha=1.0)
    #     # EuclideanShapeMetric(m=100, p=None, alpha=1.0)  # TODO
    # ],
    # [
    #     EuclideanShapeMetric(m=100, p=10, alpha=0.0),
    #     EuclideanShapeMetric(m=100, p=100, alpha=0.0),
    #     EuclideanShapeMetric(m=100, p=1000, alpha=0.0),
    #     # EuclideanShapeMetric(m=100, p=None, alpha=0.0),  # TODO
    # ],
]


for metrics in sets_of_metrics:
    if args.plot:
        fig, axs = plt.subplots(2, 2, figsize=(9, 8))

    for im, metric in enumerate(metrics):
        memo_file = args.save_dir / f"{metric.string_id()}{suffix}.pth"
        if memo_file.exists():
            print("Loading", metric.string_id())
            data = torch.load(memo_file)
        else:
            data = defaultdict(dict)

        metric_data = data[metric.string_id()]
        length_xy = np.zeros((args.num_m, args.runs))
        length_xl = np.zeros((args.num_m, args.runs))
        angle_xyz = np.zeros((args.num_m, args.runs))
        angle_xyl = np.zeros((args.num_m, args.runs))
        # Compute/load loop
        for j in trange(args.runs, desc=metric.string_id()):
            for i, sub_m in enumerate(mvals):
                metric.m = sub_m
                idx = torch.randperm(args.max_m)[:sub_m]
                sub_x, sub_y, sub_z, sub_l = data_x[idx], data_y[idx], data_z[idx], data_l[idx]
                if ("length_xy", sub_m, j) not in metric_data:
                    metric_data[("length_xy", sub_m, j)] = metric.length(*map(metric.neural_data_to_point, [sub_x, sub_y]))
                length_xy[i, j] = metric_data[("length_xy", sub_m, j)]
                if ("length_xl", sub_m, j) not in metric_data:
                    metric_data[("length_xl", sub_m, j)] = metric.length(*map(metric.neural_data_to_point, [sub_x, sub_l]))
                length_xl[i, j] = metric_data[("length_xl", sub_m, j)]
                if ("angle_xyz", sub_m, j) not in metric_data:
                    metric_data[("angle_xyz", sub_m, j)] = angle(metric, *map(metric.neural_data_to_point, [sub_x, sub_y, sub_z]))
                angle_xyz[i, j] = metric_data[("angle_xyz", sub_m, j)]
                if ("angle_xyl", sub_m, j) not in metric_data:
                    metric_data[("angle_xyl", sub_m, j)] = angle(metric, *map(metric.neural_data_to_point, [sub_x, sub_y, sub_l]))
                angle_xyl[i, j] = metric_data[("angle_xyl", sub_m, j)]
        torch.save(data, memo_file)

        if args.plot:
            c = cmap(im/10)
            mu, sigma = np.nanmean(length_xy, axis=-1), np.nanstd(length_xy, axis=-1)
            axs[0, 0].fill_between(mvals, mu-3*sigma, mu+3*sigma, color=c[:3] + (0.25,))
            axs[0, 0].plot(mvals, mu, color=c, label=".".join(metric.string_id().split(".")[:-1]))
            axs[0, 0].set_xscale("log")
            axs[0, 0].set_xlabel("m")
            axs[0, 0].set_ylabel("length(x,y)")
            axs[0, 0].legend()

            mu, sigma = np.nanmean(length_xl, axis=-1), np.nanstd(length_xl, axis=-1)
            axs[1, 0].fill_between(mvals, mu-3*sigma, mu+3*sigma, color=c[:3] + (0.25,))
            axs[1, 0].plot(mvals, mu, color=c)
            axs[1, 0].set_xscale("log")
            axs[1, 0].set_xlabel("m")
            axs[1, 0].set_ylabel("length(x,l)")

            mu, sigma = np.nanmean(angle_xyz, axis=-1), np.nanstd(angle_xyz, axis=-1)
            axs[0, 1].fill_between(mvals, mu-3*sigma, mu+3*sigma, color=c[:3] + (0.25,))
            axs[0, 1].plot(mvals, mu, color=c)
            axs[0, 1].set_xscale("log")
            axs[0, 1].set_xlabel("m")
            axs[0, 1].set_ylabel("angle(x,y,z)")

            mu, sigma = np.nanmean(angle_xyl, axis=-1), np.nanstd(angle_xyl, axis=-1)
            axs[1, 1].fill_between(mvals, mu-3*sigma, mu+3*sigma, color=c[:3] + (0.25,))
            axs[1, 1].plot(mvals, mu, color=c)
            axs[1, 1].set_xscale("log")
            axs[1, 1].set_xlabel("m")
            axs[1, 1].set_ylabel("angle(x,y,l)")

    if args.plot:
        fig.tight_layout()
        plt.show()