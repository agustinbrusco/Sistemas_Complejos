import sys
from typing import Callable
from numpy.typing import ArrayLike
import numpy as np
from findiff import FinDiff
# Importo la barra de progreso de tqdm para notebooks o para la terminal
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colormaps

TEAL_COLORS = [
    "#d8f3dc",
    "#a3dbb9",
    "#6cc297",
    "#4aa57b",
    "#3b8765",
    "#2e6a4f",
    "#214e3b",
    "#153427",
    "#081c15",
]
# Generate a matplotlib ListedColormap object
TEAL_LISTEDCMAP = colors.ListedColormap(
    colors=TEAL_COLORS, name="teal_discrete"
)
colormaps.register(TEAL_LISTEDCMAP, name="teal_discrete")
# Generate a matplotlib linear segmented colormap object
TEAL_LINEARCMAP = colors.LinearSegmentedColormap.from_list(
    colors=TEAL_COLORS, name="teal_continuous"
)
colormaps.register(TEAL_LINEARCMAP, name="teal_continuous")


def plot_colorline(
    t_vals: ArrayLike,
    x_vals: ArrayLike,
    y_vals: ArrayLike,
    cmap_name: str,
    ax: plt.Axes = None,
    max_lw: float = 3.0,
    min_lw: float = 0.5,
    min_alpha: float = 1.0,
    min_zorder: int = 40,
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
            alpha=1 - (1 - min_alpha) * norm(t),
            lw=(max_lw - (max_lw - min_lw) * norm(t)),
            zorder=min_zorder + i,
            **plot_kwargs,
        )
    return ax, colormap


def plot_mosaic_phase(
    t: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    cmap_name: str,
    max_lw: float = 3.0,
    min_lw: float = 0.5,
    min_alpha: float = 1.0,
    min_zorder: int = 40,
    **plot_kwargs,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    fig, axs_dict = plt.subplot_mosaic(
        "AAB\n" "AAC\n",
        figsize=(7 * 1.25, 4 * 1.25),
        constrained_layout=True,
    )
    _, colormap = plot_colorline(
        t,
        u,
        v,
        cmap_name,
        axs_dict["A"],
        max_lw,
        min_lw,
        min_alpha,
        min_zorder,
        **plot_kwargs,
    )
    axs_dict["A"].grid(True)
    axs_dict["A"].set_xlabel(r"$x = u$")
    axs_dict["A"].set_ylabel(r"$\dot{x} = v$")
    axs_dict["A"].set_aspect("equal")
    plt.colorbar(colormap, ax=axs_dict["A"], label="$t$", location="right")

    plot_colorline(
        t,
        t,
        u,
        cmap_name,
        axs_dict["B"],
        max_lw,
        min_lw,
        min_alpha,
        min_zorder,
        **plot_kwargs,
    )
    axs_dict["B"].grid(True)
    axs_dict["B"].set_xticklabels([])
    axs_dict["B"].yaxis.tick_right()
    axs_dict["B"].yaxis.set_label_position("right")
    axs_dict["B"].set_ylabel(r"$x = u$")

    plot_colorline(
        t,
        t,
        v,
        cmap_name,
        axs_dict["C"],
        max_lw,
        min_lw,
        min_alpha,
        min_zorder,
        **plot_kwargs,
    )
    axs_dict["C"].grid(True)
    axs_dict["C"].set_xlabel(r"$t$")
    axs_dict["C"].yaxis.tick_right()
    axs_dict["C"].yaxis.set_label_position("right")
    axs_dict["C"].set_ylabel(r"$\dot{x} = v$")
    return fig, axs_dict


def plot_fases(
    list_resultados: list[tuple[ArrayLike, ArrayLike, ArrayLike]],
    x_eq: ArrayLike,
    cmap_name: str,
    func: Callable = None,
    func_kwargs: dict[str, float] = None,
    ax: plt.Axes = None,
    max_lw: float = 3.0,
    min_lw: float = 0.5,
    min_alpha: float = 1.0,
    min_zorder: int = 40,
    **plot_kwargs,
) -> tuple[plt.Axes, plt.cm.ScalarMappable]:
    if ax is None:
        _, ax = plt.subplots(1, 1,)
    for resultados in tqdm(
        list_resultados, total=len(list_resultados), desc="Colorline Plots", leave=False
    ):
        t, u, v = resultados
        ax, colormap = plot_colorline(
            t, u, v, cmap_name, ax, max_lw, min_lw, min_alpha, min_zorder, **plot_kwargs
        )
    if x_eq is not None:
        ax.plot(x_eq, [0 for val in x_eq], ".", ms=max_lw, c="k")
    if func is not None:
        stream_color = plt.colormaps[cmap_name](0.5)
        u_lims = ax.get_xlim()
        ax.set_xlim(u_lims)
        v_lims = ax.get_ylim()
        ax.set_ylim(v_lims)
        U, V = np.meshgrid(
            np.linspace(*u_lims, 100), np.linspace(*v_lims, 100), indexing="xy"
        )
        if func_kwargs is None:
            func_kwargs = {}
        dU, dV = func(0, (U, V), **func_kwargs)
        ax.streamplot(
            U, V, dU, dV,
            color=stream_color, density=2., linewidth=min_lw, arrowsize=min_lw
        )
    ax.grid()
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\dot{x}$")
    return ax, colormap


# Funciones redudantes con el notebook
def runge_kutta_4_step(
    f: Callable,
    x_i: ArrayLike,
    t_i: float,
    dt: float,
    **kwargs,
) -> ArrayLike:
    k_array = np.empty((4, x_i.size))
    k_array[0] = f(t_i, x_i, **kwargs)
    k_array[1] = f(t_i + dt * 0.5, x_i + dt * 0.5 * k_array[0], **kwargs)
    k_array[2] = f(t_i + dt * 0.5, x_i + dt * 0.5 * k_array[1], **kwargs)
    k_array[3] = f(t_i + dt, x_i + dt * k_array[2], **kwargs)
    a_vec = np.array([1, 2, 2, 1]) / 6
    a_vec = a_vec.reshape((4, 1))
    return x_i + dt * np.sum(a_vec * k_array, axis=0)


def runge_kutta_4(
    f: Callable,
    x_0: ArrayLike,
    dt: float,
    steps: int,
    **kwargs,
) -> ArrayLike:
    x_vals = np.zeros((1 + steps, x_0.size))
    x_vals[0] = x_0
    for i in range(steps):
        x_vals[i + 1] = runge_kutta_4_step(f, x_vals[i], i * dt, dt, **kwargs)
    return x_vals


def balance(
    energia: Callable,
    variacion_teorica: Callable,
    u: ArrayLike,
    v: ArrayLike,
    dt: float,
    **kwargs,
) -> ArrayLike:
    """
    Esta función recibe u (x), v (dx/dt), parametros extra y
    las funciónes de la energía y su derivada. Devuelve un vector
    de balance que tiene la diferencia de la energía del sistema
    en cada tiempo con respecto al valor teórico normalizado por
    el valor inicial de la energía.
    """
    d_dt = FinDiff(0, dt, acc=6)
    E = energia(u, v, **kwargs)
    dEdt_numerico = d_dt(E)
    dEdt_teorico = variacion_teorica(u, v, **kwargs)
    return dt * (dEdt_numerico - dEdt_teorico) / E[0]
