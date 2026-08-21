import numpy as np
import matplotlib.pyplot as plt

from typing import List, Optional
from numpy.typing import NDArray


def logpost_plot(
    log_post: NDArray,
    model_names: List[str],
    save_path: Optional[str] = None,
    filename: str = "logpost",
    file_ext: str = "svg",
    roi: Optional[str] = None,
    bayes_method: Optional[str] = None,
    extra_title: Optional[str] = None,
):
    """
    Display and optionally save a plot of the posterior over models for a given
    set of brain data.

    Parameters
    ----------

    log_post: NDArray
        Log posterior results from BayesCompare analysis. Expected shape is
        (n_voxels, n_models)

    model_names: List of str
        List containing the model names in the order as they are indexed in
        log_post

    save_path: str or None, default None
        Path to save the figure to

    filename: str, default="logpost"
        Only used if save_path is provided

    file_ext: str, default="svg"
        Used in case of saving

    roi: str or None, default=None
        Name of the area(s) that the data representa. Used for filename in case
        of saving

    bayes_method: str or None, default=None
        Optional string for different analysis methods in filename. Only used
        when saving

    extra_title: str or None, default=None
        String attached at the end of the figure title, useful to include things
        like subject number, ROI name, etc.
    """
    n_models = len(model_names)
    plt.figure(figsize=(min(15, n_models * 3), n_models * 2))
    plt.axes((0.1, 0.1, 0.8, 0.55))
    plt.boxplot(log_post, whis=(0, 100))
    plt.xlim(0, len(model_names) + 1)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.xlabel("Models", fontsize=15)
    plt.xticks(range(1, len(model_names) + 1), model_names, rotation=90)
    plt.ylabel("LogPost", fontsize=15)

    m_win, counts = np.unique(np.argmax(log_post, 1), return_counts=True)
    plt.axes((0.1, 0.7, 0.8, 0.2))
    plt.bar(m_win + 1, counts, color="k")
    plt.xlim(0, len(model_names) + 1)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.xticks([])
    plt.ylabel("N voxels", fontsize=15)

    plot_title = "Log posterior and winning model"
    if extra_title:
        plot_title += f" in {extra_title}"
    plt.title(plot_title, fontsize=15)

    if bayes_method:
        filename = f"{filename}_{bayes_method}"
    if save_path:
        plt.savefig(f"{save_path}/{filename}_{roi}.{file_ext}", bbox_inches="tight")

    plt.show()
