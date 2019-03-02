import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.interpolate
from sister.datasets import ColorUtils


def loadData(path):
    data = np.loadtxt(path, skiprows=1)
    bs = np.array([0.002, 0.01, 0.025, 0.05, 0.1, 0.15])
    hs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55])
    return data, hs, bs


bs = np.array([0.002, 0.01, 0.025, 0.05, 0.1, 0.15])
hs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55])

f = open('/home/daniele/data/datasets/sister/v1/multi_planes/planes_data.txt', 'r')
lines = f.readlines()

plt.figure(0)
color_index = 0
for i in range(0, len(lines), 2):
    sx = lines[i]
    print(lines[i])
    print(lines[i + 1])
    x = np.fromstring(lines[i], sep=' ')
    y = np.fromstring(lines[i + 1], sep=' ')

    f = scipy.interpolate.interp1d(x, y, kind='cubic')
    newx = np.linspace(x[0], x[-1], 25, endpoint=True)
    newy = f(newx)
    newy = scipy.signal.medfilt(newy, 15)

    plt.plot(newx, newy, color=ColorUtils.getColorByIndex(color_index , fmt='rgbf'))
    color_index+=1
color_index = 0

#plt.title("Multi")
plt.legend(['Baseline {} m'.format(bs[i]) for i in range(len(bs))])
plt.xlabel('distance [m]')
plt.ylabel('MAE [m]')
plt.yticks(np.arange(0, 1.5, 0.001))
plt.xticks(np.arange(0, 1.5, 0.05))
plt.grid()
axes = plt.gca()
axes.set_ylim([0, 0.01])
axes.set_xlim([0, 0.55])

min_zs = [0.01, 0.05, 0.1, 0.15, 0.3, 0.4]
fig = plt.figure(1, figsize=(8, 3))

full_dz = []
for i, b in enumerate(bs):
    j = 0
    while hs[j] < min_zs[i]:
        j += 1
    new_hs = hs[j:]

    whole_hs = hs.copy()
    whole_dz = (whole_hs ** 2) / (b * (20 / 0.006))
    whole_dz[whole_hs < min_zs[i]] = 1000

    full_dz.append(whole_dz)
    print("WHOLE", full_dz)
    dz = (new_hs ** 2) / (b * (20 / 0.006))
    plt.plot(new_hs, dz, color=ColorUtils.getColorByIndex(color_index , fmt='rgbf'))
    color_index+=1

# full_dz = np.array(full_dz)
# full_dz = np.min(full_dz,0)
# print("FULL",full_dz)
# plt.plot(hs, full_dz.ravel(),linewidth=5,  alpha=0.8)

#plt.title("MULTIBASELINE")
plt.legend(['Baseline {} m'.format(bs[i]) for i in range(len(bs))] + ['Theoretical Minimum'])
plt.xlabel('distance [m]')
plt.ylabel('Depth error [m]')
plt.grid()
plt.yticks(np.arange(0, 1.5, 0.001))
plt.xticks(np.arange(0, 1.5, 0.05))
axes = plt.gca()
axes.set_ylim([0, 0.003])
axes.set_xlim([0, 0.55])
fig.subplots_adjust(bottom=0.2)
plt.show()
