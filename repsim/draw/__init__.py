"""
A module to manage the drawing of representation paths.

Uses matplotlib.

"""

from typing import Literal
import matplotlib.pyplot as plt
import numpy as np


def clump_diagram(
    embeddings,
    layers_per_model,
    labels=None,
    show_cumulative_row=True,
    color_by_layer=False,
    cmap="Blues_r",
):
    """
    Plot a clump diagram.

    Gallery:
    ![62ae215b-4abd-4921-bd6d-e512c8c50350](https://user-images.githubusercontent.com/693511/160645577-40d02384-fa48-4252-8033-93a7aa875cbd.png)

    Arguments:
        embeddings (tensor): A list of points to plot.
        layers_per_model (int): The number of layers per model. If None, all
            layers are interpreted as one continuous path.
        labels (list[str]): A list of labels for each point. If specified, the
            labels will be included in the plot y-axis.
        show_cumulative_row (bool): If True, the cumulative row will be shown
            at the bottom of the plot.
        color_by_layer (bool): If True, the lines will be colored by their
            respective layer.
        cmap (str): The colormap to use.

    """
    layers_per_model = layers_per_model or len(embeddings)
    num_models = len(embeddings) // layers_per_model

    cmap = plt.get_cmap(cmap)
    colors = [
        cmap(i / num_models) if color_by_layer else cmap(1) for i in range(num_models)
    ]

    for model in range(num_models):
        this_model_layers = embeddings[
            model * layers_per_model : (model + 1) * layers_per_model
        ]
        distances = [
            np.linalg.norm(i - j)
            for i, j in zip(this_model_layers, this_model_layers[1:])
        ]
        # Draw a straight line at y=model with vertical ticks
        # at each distance
        plt.plot(
            [0, np.sum(distances)],
            [model, model],
            color="black",  # if color_by_layer else colors[model],
            linewidth=2,
        )
        if show_cumulative_row:
            plt.plot(
                [0, np.sum(distances)],
                [-1, -1],
                color="black",
                linewidth=1.9,
            )
        agg_dist = 0
        for i, d in enumerate(distances):
            # Draw a vertical line at the distance
            plt.plot(
                [agg_dist, agg_dist],
                [model - 0.5, model + 0.5],
                color="black" if not color_by_layer else colors[model],
                linewidth=0.75,
            )
            if show_cumulative_row:
                plt.plot(
                    [agg_dist, agg_dist],
                    [-1 - 0.25, -1 + 0.25],
                    color="black" if not color_by_layer else colors[model],
                    linewidth=0.75,
                    alpha=0.85,
                )
            agg_dist += d

    if labels:
        if show_cumulative_row:
            plt.yticks(range(-1, num_models), ["Cumulative", *labels])
        else:
            plt.yticks(range(num_models), labels)
    plt.xticks([], [])


def arrow_plot(
    embeddings,
    layers_per_model: int = None,
    destination: list[float] = None,
    labels: list[str] = None,
    cmap: str = "Blues_r",
    arrow_type: Literal["fancy", "simple"] = "simple",
):
    """
    Draw an arrow plot of the given paths.

    Gallery:
    ![a7988120-ea4c-4642-be1c-a557a8459526](https://user-images.githubusercontent.com/693511/160639037-8cfbec26-4d80-47aa-81a2-a5f5d98997e7.png)
    ![af3f32eb-fb0f-4342-9604-9977e5952906](https://user-images.githubusercontent.com/693511/160639041-60e2f26c-a8f4-44b7-b0b7-5ae6041d9127.png)

    Arguments:
        pts (tensor): A list of points to plot. Each point should be have at
            least two coordinates. If more coordinates are given, only the
            first two are used.
        layers_per_model (int): The number of layers per model. If None, all
            layers are interpreted as one continuous path. If an integer, the
            layers are interpreted as paths of length `layers_per_model`.
        destination (list[float]): The coordinates of the destination point. If
            specified, this will be plotted as a red star.
        labels (list[str]): A list of labels for each point. If specified, the
            labels will be included in the plot legend. If not specified, the
            legend will not be rendered.
        cmap (str): The name of the colormap to use.
        arrow_type (str): The type of arrow to use. Can be "fancy" or "simple".

    """
    cmap = plt.get_cmap(cmap)
    layers_per_model = layers_per_model or len(embeddings)
    model_count = len(embeddings) / layers_per_model
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(embeddings, embeddings[1:])):
        if (i + 1) % layers_per_model:
            c = i // layers_per_model
            if arrow_type == "simple":
                plt.arrow(
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                    # color=plt.cm.Blues_r( c / model_count),
                    color=cmap(c / model_count),
                    width=0.01,
                    head_width=1,
                    label=f"{labels[c]}"
                    if labels and ((i + 1) % layers_per_model == 1)
                    else None,
                    length_includes_head=True,
                )
            elif arrow_type == "fancy":
                plt.annotate(
                    "",
                    xytext=(x1, y1),
                    xy=(x2, y2),
                    arrowprops=dict(
                        arrowstyle="->",
                        # color=f"C{c}",
                        color=cmap(c / model_count),
                        lw=1,
                    ),
                )

    if destination:
        # plot a star at the destination point
        plt.scatter(
            destination[0],
            destination[1],
            marker="*",
            c="red",
            s=50,
            label=None if labels is None else "Target",
        )

    if labels:
        plt.legend()
    plt.xlim((embeddings[:, 0].min()), (embeddings[:, 0].max()))
    plt.ylim((embeddings[:, 1].min()), (embeddings[:, 1].max()))


# TODO: angle hist
