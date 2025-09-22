from scipy.differentiate import jacobian, hessian

import numpy as np


def f(v) -> float:
	return v[0] + v[1]


if __name__ == '__main__':
	res = jacobian(f, [1, 1, 1]).df
	print(res)