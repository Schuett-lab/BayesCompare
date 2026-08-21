import numpy as np
import matplotlib.pyplot as plt


def logpost_plot(
    log_post,
    model_names,
    roi,
    save_path=None,
    filename="logpost",
    file_ext="svg",
    bayes_method=None,
    extra_title=None,
):
    n_models = len(model_names)
    plt.figure(figsize=(min(15, n_models * 3), n_models * 2))
    plt.axes((0.1, 0.1, 0.8, 0.55))
    plt.boxplot(log_post, whis=[0, 100])
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
