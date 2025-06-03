import numpy as np
from matplotlib import pyplot as plt

# network 1
x1 = np.array([[1.5, 0.5], [1, 1]])
Sigma1 = x1 @ x1.T
norm1 = np.mean(np.diag(Sigma1))
Sigma1 /= norm1
Sigma1 = 0.99 * Sigma1 + 0.01 * np.eye(2)
Prec1 = np.linalg.inv(Sigma1)
# network 2
x2 = np.array([[-1.25 / 2, -0.25], [0.5, 1]])
Sigma2 = x2 @ x2.T
norm2 = np.mean(np.diag(Sigma2))
Sigma2 /= norm2
Sigma2 = 0.99 * Sigma2 + 0.01 * np.eye(2)
Prec2 = np.linalg.inv(Sigma2)


# plot network 1
plt.figure()
plt.arrow(0, 0, x1[0, 0], x1[0, 1], width=0.01, length_includes_head=True)
plt.arrow(0, 0, x1[1, 0], x1[1, 1], width=0.01, length_includes_head=True)
plt.arrow(-1.5, 0, 3, 0, width=0.01, length_includes_head=True)
plt.arrow(0, -1.5, 0, 3, width=0.01, length_includes_head=True)
plt.axis("square")
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.gca().set_axis_off()
plt.savefig("figures/network1_act.pdf")

# plot network 2
plt.figure()
plt.arrow(0, 0, x2[0, 0], x2[0, 1], width=0.01, length_includes_head=True)
plt.arrow(0, 0, x2[1, 0], x2[1, 1], width=0.01, length_includes_head=True)
plt.arrow(-1.5, 0, 3, 0, width=0.01, length_includes_head=True)
plt.arrow(0, -1.5, 0, 3, width=0.01, length_includes_head=True)
plt.axis("square")
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.gca().set_axis_off()
plt.savefig("figures/network2_act.pdf")


x_plot = np.linspace(-3, 3, 1000)
xx = np.meshgrid(x_plot, x_plot)
xx = np.stack(xx)
pdf1 = np.exp(-0.5 * np.einsum("ijk, il, ljk->jk", xx, Prec1, xx))
pdf2 = np.exp(-0.5 * np.einsum("ijk, il, ljk->jk", xx, Prec2, xx))

x_cond = np.where(x_plot > 1)[0][0]

# pdfs = np.stack((pdf1, np.zeros_like(pdf1), pdf2))
# plt.imshow(np.transpose(pdfs, (1, 2, 0)), extent=(-3, 3, -3, 3))
plt.figure()
plt.arrow(
    0, 500, 1000, 0, width=1,
    head_width=15, head_length=30, length_includes_head=True,
    color="black")
plt.arrow(
    500, 0, 0, 1000, width=1,
    head_width=15, head_length=30, length_includes_head=True,
    color="black")
plt.contour(pdf1, colors="red")
plt.contour(pdf2, colors="blue")
plt.plot([x_cond, x_cond], [0, 999], 'k--')
plt.axis("square")
plt.tick_params('both', direction="out")
plt.gca().set_axis_off()
plt.savefig("figures/pdfcontours.pdf")


pdf1_1d = pdf1[:, x_cond].copy()
pdf1_1d /= np.sum(pdf1_1d)
pdf2_1d = pdf2[:, x_cond].copy()
pdf2_1d /= np.sum(pdf2_1d)


plt.figure()
plt.plot(x_plot, pdf1_1d, "red")
plt.plot(x_plot, pdf2_1d, "blue")
plt.savefig("figures/dens1D.pdf")

plt.close('all')
