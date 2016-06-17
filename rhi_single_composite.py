rhi8 = xpol.get_data(8, 'RHI', 180, homedir=homedir, index=53)
rhi9 = xpol.get_data(9, 'RHI', 180, homedir=homedir, index=53)
rhi10 = xpol.get_data(10, 'RHI', 180, homedir=homedir, index=53)
rhi11 = xpol.get_data(11, 'RHI', 180, homedir=homedir, index=29)
rhi12 = xpol.get_data(12, 'RHI', 6, homedir=homedir, index=53)
rhi13 = xpol.get_data(13, 'RHI', 180, homedir=homedir, index=53)
rhi14 = xpol.get_data(14, 'RHI', 180, homedir=homedir, index=53)

za8 = rhi8['ZA'].iloc[0]
za9 = rhi9['ZA'].iloc[0]
za10 = rhi10['ZA'].iloc[0]
za11 = rhi11['ZA'].iloc[0]
za12 = rhi12['ZA'].iloc[0]
za13 = rhi13['ZA'].iloc[0]
za14 = rhi14['ZA'].iloc[0]
za = [za8, za9, za10, za11, za12, za13, za14]

midp = np.array([286, 286, 286, 215, 157, 215, 215])
offset = 290-midp
comp = np.zeros((61, 505))
for n, z in enumerate(za):
    size = z.shape[1]
    end = size+offset[n]
    z[np.isnan(z)] = 0
    comp[:, offset[n]:size+offset[n]] += z

plt.figure()
im = plt.imshow(comp, vmin=-10, vmax=45,
                interpolation='none', origin='lower', aspect='auto')
plt.colorbar(im)
plt.show(block=False)