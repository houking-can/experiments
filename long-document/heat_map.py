import matplotlib.pyplot as plt
def plot_heatmap(a, inwords, outwords):
    # weight matrix
    # plot 2 heatmap
    # refer to document from matplotlib
    # https://matplotlib.org/gallery/axes_grid1/demo_axes_divider.html#sphx-glr-gallery-axes-grid1-demo-axes-divider-py
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.figure(figsize=(27, 15))
    ax = plt.subplot()
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="2%", pad=0.2)
    fig1 = ax.get_figure()
    fig1.add_axes(ax_cb)
    im = ax.imshow(a, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
    ax.set_yticks(range(len(outwords)))
    ax.set_yticklabels(outwords, fontsize=20)
    ax.set_xticks(range(len(inwords)))
    ax.set_xticklabels(inwords)
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)
        label.set_fontsize(20)
    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    ax_cb.yaxis.set_tick_params(labelright=True)
    ax.set_title('Attention Heatmap (out-in)', fontsize=20)
    plt.savefig(f"./image/attention_heatmap.png")