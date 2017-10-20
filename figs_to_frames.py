
import matplotlib.pyplot as plt


curr_pos = 0


def figs_to_frames(da):

    def plot(ops_cf):

        da.isel(time=curr_pos).plot.contourf(**ops_cf)
        # lonlims = [arr1.lon.min(), arr1.lon.max()]
        # latlims = [arr1.lat.min(), arr1.lat.max()]
        # ax.set_xlim(lonlims)
        # ax.set_ylim(latlims)
        # plt.tight_layout()


    def key_event(e):

        global curr_pos

        if e.key == "right":
            curr_pos = curr_pos + 1
        elif e.key == "left":
            curr_pos = curr_pos - 1
        else:
            return

        curr_pos = curr_pos % len_plots
        print 'asdasd'
        ax.cla()
        ops_cf['add_colorbar'] = False
        plot(ops_cf)
        fig.canvas.draw()
        plt.show()

    # figh = 12
    # figw = 6
    fig = plt.figure()
    ax = plt.axes()
    fig.add_axes(ax)
    fig.canvas.mpl_connect('key_press_event', key_event)

    ops_cf = dict(x='x', y='z',
                  # levels=range(200, 1100, 100),
                  # extend='max',
                  # cmap='RdYlBu_r',
                  # transform=ccrs.PlateCarree(),
                  # subplot_kws={'projection': ccrs.PlateCarree()},
                  add_colorbar=True,
                  ax=ax
                  )

    # ops_ct = dict(transform=ccrs.PlateCarree(),
    #               colors='k',
    #               ax=ax,
    #               )

    len_plots = da.time.size
    plot(ops_cf)
    plt.show()

