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


if __name__ == "__main__":
    # Smoke test: synthetic decaying spectrum to eyeball that the plot renders.
    import matplotlib

    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt

    synthetic = np.array([100.0, 30.0, 9.0, 3.0, 1.0, 0.3, 0.1, 0.03])
    fig, _ = build_scree_plot_figure(synthetic, title="Synthetic scree plot")
    # In the smoke test we just want to see the figure — in production the
    # MatplotlibDisplayWidget owns the canvas instead.
    plt.figure(fig.number)
    plt.show()
