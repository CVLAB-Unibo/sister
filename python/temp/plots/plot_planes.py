import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


def loadData(path):
    data = np.loadtxt(path, skiprows=1)
    bs = np.array([0.002, 0.01, 0.025, 0.05, 0.1, 0.15])
    hs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,0.5,0.55])
    return data, hs, bs


data, hs, bs = loadData('/tmp/plane_refine2.txt')
print(data)

f = open('/tmp/planetemp.txt','a')
n = 6
min_d = np.min(data[:, 3])
max_d = np.max(data[:, 3])
for i in range(n):
    d = data[i::6, 2]
    #d = np.clip(d, 0, 0.1)
    d = scipy.signal.medfilt(d, 3)

    x = hs
    y = d
    print(i)
    print(x)
    print(y)
    np.savetxt(f, x.reshape((1,-1)), fmt='%10.5f')
    np.savetxt(f, y.reshape((1,-1)), fmt='%10.5f')
    plt.plot(x, y)

    #

    # x = hs
    # y = (hs**2) / (bs[i] * (500/0.006))
    # plt.plot(x, y,dashes=[30, 5, 10, 5])


plt.title("ARDUINO - DISTANCE vs BASELINE")
plt.legend(['Baseline {} m'.format(bs[i]) for i in range(n)])
plt.xlabel('distance [m]')
plt.ylabel('RMSE')
plt.show()

# for i in range(n):
#     d = data[i::6, 1]
#     #d = np.clip(d, 0, 0.1)
#     d = scipy.signal.medfilt(d, 3)
#
#     x = hs[i:]
#     y = d[i:]
#     plt.plot(x, y)

plt.figure(1)
for b in bs:

    dz = (hs**2) / (b * (350/0.006))
    print(dz)
    plt.plot(hs,dz)

plt.show()