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
