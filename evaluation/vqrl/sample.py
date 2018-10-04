import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,2*np.pi,100)
y = np.sin(x)

ax = plt.subplot(1,1,1)
ax.spines['bottom'].set_linewidth(5)
plt.plot(x,y,'r')
plt.show()
