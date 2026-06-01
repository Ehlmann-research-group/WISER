"""Shared matplotlib helper for rendering scree plots of eigenvalues.

Used by PCA and MNF past-runs viewers (see :mod:`wiser.gui.pca_history` and
:mod:`wiser.gui.mnf_history`).  The eigenvalues passed in may be a subset of
the full spectrum (e.g. sklearn PCA only retains ``num_components`` of them);
that is acceptable — we just plot what we have.

The caller is responsible for displaying the returned ``(figure, axes)`` via
:meth:`wiser.gui.app_state.ApplicationState.show_matplotlib_display_widget`,
which owns the window and cleans it up on close.
"""

from typing import Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def build_scree_plot_figure(
    eigenvalues: np.ndarray,
    *,
    title: str,
    x_label: str = "Component index",
    y_label: str = "Eigenvalue",
) -> Tuple[Figure, Axes]:
    """Build a scree plot of ``eigenvalues`` vs 1-indexed component position.

    Args:
        eigenvalues: 1-D array of eigenvalues, ordered descending (the
            convention used by both the sklearn PCA result and WISER's
            ``EigenDecompositionStage``).  May be a subset of the full
            spectrum — the plot just renders what is given.
        title: Plot title (becomes the matplotlib axes title; the caller
            typically also uses it as the window title).
        x_label, y_label: Axis labels.

    Returns:
        A ``(Figure, Axes)`` pair.  The figure is not displayed — pass it to
        ``ApplicationState.show_matplotlib_display_widget`` so the window is
        properly owned and torn down on close.
    """
    values = np.asarray(eigenvalues, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("Cannot build a scree plot from an empty eigenvalue array.")

    component_indices = np.arange(1, values.size + 1)

    figure = Figure(figsize=(6.0, 4.0))
    # Tight layout keeps tick labels and title from being clipped when the
    # window is small.
    figure.set_tight_layout(True)
    axes = figure.add_subplot(111)

    axes.plot(component_indices, values, marker="o", linestyle="-", linewidth=1.2)

    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)

    # Force integer ticks on the x-axis — fractional component indices make
    # no sense.  For small N (<= 30) show every index; for larger N let
    # matplotlib decide but still constrain to integers.
    if values.size <= 30:
        axes.set_xticks(component_indices)

    # Log scale is the right default — eigenvalue spectra typically span many
    # orders of magnitude.  Matplotlib silently drops non-positive values in
    # log mode, which is acceptable: zero eigenvalues mean the corresponding
    # direction is degenerate and there's nothing meaningful to plot.  Fall
    # back to linear when *every* value is non-positive (pathological but
    # possible) so the plot still renders.
    if np.any(values > 0):
        axes.set_yscale("log")
    axes.grid(True, which="both", linestyle="--", alpha=0.4)

    return figure, axes


def attach_scree_click_inspector(
    figure: Figure,
    axes: Axes,
    eigenvalues: np.ndarray,
) -> None:
    """Make the scree plot click-inspectable.

    On every click inside ``axes``, snaps to the nearest component along the
    x-axis (1-indexed) and writes ``PC{idx}: {value}`` into a small readout
    pinned to the top-left corner of the plot.  Subsequent clicks update the
    same readout in place rather than stacking annotations.

    Must be called *after* the figure has been wrapped in its final Qt
    canvas (i.e. after
    :meth:`wiser.gui.app_state.ApplicationState.show_matplotlib_display_widget`).
    Constructing a new ``FigureCanvas`` around the figure replaces
    ``figure.canvas`` and silently discards any earlier ``mpl_connect``
    bindings, so connecting too early is a no-op at runtime.
    """
    values = np.asarray(eigenvalues, dtype=np.float64).ravel()
    if values.size == 0:
        return

    # Pinned readout in axes-fraction coordinates so it stays in the top-left
    # corner regardless of pan/zoom.  Created hidden so the box only appears
    # after the user actually clicks something.
    readout = axes.text(
        0.02,
        0.98,
        "",
        transform=axes.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9),
    )
    readout.set_visible(False)

    # Dotted crosshair through the snapped point — matches the style used by
    # SpectrumPlotGeneric's point selection (see CROSSHAIR_WIDTH in
    # spectrum_plot.py).  axvline / axhline span the full axes regardless of
    # data range so the lines extend across the whole plot.  Initial x/y are
    # placeholders; the click handler moves them.
    vline = axes.axvline(x=0, linestyle="dotted", color="black", linewidth=0.5, zorder=1)
    hline = axes.axhline(y=0, linestyle="dotted", color="black", linewidth=0.5, zorder=1)
    vline.set_visible(False)
    hline.set_visible(False)

    def _on_click(event):
        # Ignore clicks outside the scree axes (toolbar, figure margins,
        # other axes) — event.xdata is None in those cases too, but the
        # axes check is the more readable guard.
        if event.inaxes is not axes or event.xdata is None:
            return
        # Snap to the nearest 1-indexed component, clamped to the available
        # range so clicks outside the data still produce a sensible label.
        idx = int(round(event.xdata)) - 1
        idx = max(0, min(values.size - 1, idx))
        x = idx + 1
        y = values[idx]
        readout.set_text(f"PC{x}: {y:.4g}")
        readout.set_visible(True)
        # set_xdata on an axvline expects a 2-element sequence (the line's
        # two endpoints share the same x); same for axhline with y.
        vline.set_xdata([x, x])
        hline.set_ydata([y, y])
        vline.set_visible(True)
        hline.set_visible(True)
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("button_press_event", _on_click)


if __name__ == "__main__":
    # Smoke test: synthetic decaying spectrum to eyeball that the plot renders.
    import matplotlib

    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt

    synthetic = np.array([100.0, 30.0, 9.0, 3.0, 1.0, 0.3, 0.1, 0.03])
    fig, ax = build_scree_plot_figure(synthetic, title="Synthetic scree plot")
    # pyplot.show() will give the figure a real canvas; attach the inspector
    # after that so the click handler is bound to the live canvas.
    plt.figure(fig.number)
    attach_scree_click_inspector(fig, ax, synthetic)
    plt.show()
