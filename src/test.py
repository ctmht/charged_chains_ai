from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':
	X = np.linspace(0, 1, 100)
	plt.plot(X, X+1)
	plt.show()