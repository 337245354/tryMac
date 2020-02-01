import numpy as np
import matplotlib.pyplot as plt
def logp2(x):
    y = -np.log(1-x)
    return y
plot_x = np.linspace(0, 0.99, 50) #取0.99避免除数为0
plot_y = logp2(plot_x)
plt.plot(plot_x, plot_y)
plt.show()