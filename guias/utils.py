from numpy.typing import ArrayLike
import matplotlib.pyplot as plt


def plot_colorline(
    t_vals: ArrayLike,
    x_vals: ArrayLike,
    y_vals: ArrayLike,
    cmap_name: str,
    ax: plt.Axes = None,
    **plot_kwargs,
) -> tuple[plt.Axes, plt.cm.ScalarMappable]:
    if ax is None:
        _, ax = plt.subplots(1, 1)
    norm = plt.Normalize(
        t_vals.min(),
        t_vals.max(),
    )
    colormap = plt.cm.ScalarMappable(norm, cmap_name)
    for i, t in enumerate(t_vals[:-1]):
        ax.plot(
            [x_vals[i], x_vals[i + 1]],
            [y_vals[i], y_vals[i + 1]],
            c=colormap.to_rgba(t),
            alpha=(1 - norm(t) + 0.5) / 1.5,
            lw=(3 - 2 * norm(t)),
            **plot_kwargs,
        )
    return ax, colormap


def plot_mosaic_phase(
    t: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    cmap_name: str,
    **plot_kwargs,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    fig, axs_dict = plt.subplot_mosaic(
        "AAB\n" "AAC\n",
        figsize=(7 * 1.25, 4 * 1.25),
        constrained_layout=True,
    )
    _, colormap = plot_colorline(t, u, v, cmap_name, ax=axs_dict["A"], **plot_kwargs)
    axs_dict["A"].grid(True)
    axs_dict["A"].set_xlabel(r"$x = u$")
    axs_dict["A"].set_ylabel(r"$\dot{x} = v$")
    axs_dict["A"].set_aspect("equal")
    plt.colorbar(colormap, ax=axs_dict["A"], label="$t$", location="right")

    plot_colorline(t, t, u, cmap_name, ax=axs_dict["B"], **plot_kwargs)
    axs_dict["B"].grid(True)
    axs_dict["B"].set_xticklabels([])
    axs_dict["B"].yaxis.tick_right()
    axs_dict["B"].yaxis.set_label_position("right")
    axs_dict["B"].set_ylabel(
        r"$x = u$",
    )

    plot_colorline(t, t, v, cmap_name, ax=axs_dict["C"], **plot_kwargs)
    axs_dict["C"].grid(True)
    axs_dict["C"].set_xlabel(r"$t$")
    axs_dict["C"].yaxis.tick_right()
    axs_dict["C"].yaxis.set_label_position("right")
    axs_dict["C"].set_ylabel(r"$\dot{x} = v$")
    return fig, axs_dict
