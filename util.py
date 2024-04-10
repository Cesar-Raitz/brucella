import winsound

def beep():
	winsound.Beep(1440, 200)
	

#%%
def overlap(a0, a1, b0, b1) -> float:
	if a0 > a1: a0, a1 = a1, a0
	if b0 > b1: b0, b1 = b1, b0
	if a0 >= b1 or b0 >= a1:
		return 0
	elif b1 > a1:
		return a1 - max(a0, b0)
	else:
		return b1 - max(a0, b0)


if __name__ == "__main__":
	def test_overlap(args, expected):
		res = overlap(*args)
		print(f"overlap{args} == {expected} ", end='')
		if abs(res - expected) < 1e-6:
			print("Ok")
		else:
			print(f"FAILED! (got {res})")
	
	for args, expected in [
			((1, 3, 4, 5), 0), # b to the right of a
			((1, 3, 3, 4), 0), # b to the right of a (sharing one limit)
			((1, 3, 2, 5), 1), # b partially inside a (to the right)
			((1, 3, 2, 3), 1), # b inside a (sharing one limit)
			((1, 3, 1, 2), 1), # b inside a (sharing one limit)
			((0, 3, 1, 2), 1), # b inside a (fully)

			((1, 3, 1, 3), 2), # a equals b

			((1, 2, 0, 3), 1), # a inside b (fully)
			((2, 3, 0, 3), 1), # a inside b (sharing one limit)
			((0, 1, 0, 3), 1), # a inside b (sharing one limit)
			((2, 5, 0, 3), 1), # a partially inside b (to the right)
			((1, 3, 0, 1), 0), # a to the right of b (sharing one limit)
			((1, 3, -1, 0), 0) # b to the left of a
			]:
		test_overlap(args, expected)
	print("All Tests Done!")


#%%
import matplotlib.pyplot as plt

class Curve:
	curr_index = 0
	color_table = ["deepskyblue", "darkorange", "blueviolet", "blue",
					   "crimson", "dimgrey", "mediumturquoise", "deeppink"]

	def __init__(self, x, y=None, e=None, label=None, color=None):
		if isinstance(x, dict):
			label = x.get('label', label)
			color = x.get('color', color)
			e = x['e']
			y = x['y']
			x = x['x']
		
		self._x = x
		self._y = y
		self._e = e
		self._label = label

		if color:
			self._color = color
		else:
			self._color = Curve.color_table[Curve.curr_index]
			if Curve.curr_index < len(Curve.color_table)-1:
				Curve.curr_index += 1
			else:
				Curve.curr_index = 0
		
	def plot_error(self, ax: plt.Axes=None):
		if ax is None: ax = plt.gca()
		x = self._x
		y1 = self._y - self._e
		y2 = self._y + self._e
		ax.fill_between(x, y1, y2, color=self._color, alpha=0.25)
		ax.plot(x, y1, color=self._color, alpha=0.3)
		ax.plot(x, y2, color=self._color, alpha=0.3)
	
	def plot_curve(self, ax: plt.Axes=None):
		if ax is None: ax = plt.gca()
		ax.plot(self._x, self._y, color=self._color, label=self._label)


if __name__ == "__main__":
	import numpy as np
	x = np.linspace(0, 10)
	e = np.ones(len(x))
	c1 = Curve(x, -0.2*x**2 + x, e, "Curve 1")
	c2 = Curve(x, 0.2*x**2 - x - 10, e, "Curve 2")
	c1.plot_error()
	c1.plot_curve()
	c2.plot_error()
	c2.plot_curve()
	plt.legend(loc=10)


#%%
def progress_bar(pos, max_pos, size=32):
	size -= 2
	f = pos/max_pos
	n = int(f*size + 0.5)
	text = "\r[" + '='*n + ' '*(size-n) + f"] {100*f:.0f}%"
	print(text, end='')


if __name__ == "__main__":
	progress_bar(30, 100)
	progress_bar(50, 100)
	progress_bar(80, 100)