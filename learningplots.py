
import matplotlib.pyplot as plt
plt.style.use("seaborn-white")
import numpy as np

ax1 = plt.axes()
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])

#These numbers represent [left, bottom, width, height]
#https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html

fig = plt.figure()

ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                   xticklabels = [], ylim=(-1.2, 1.2))

ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
                   ylim=(-1.2, 1.2))

x = np.linspace(0,10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x))

#-----------------------------------------------
#Simple Grids of Subplots

for i in range(1,7):
    plt.subplot(2,3,i)
    plt.text(0.5,0.5, str((2,3,i)), fontsize =18, ha='center')

#

