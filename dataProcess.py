import os
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
read_file_path = "./result/att48_SA_result.txt"
# write_file_path = "./result/att48_result.txt"
# with open(read_file_path, "r") as f_1:
# 	with open(write_file_path, "w") as f_2:
# 		for each in f_1.readlines():
# 			pre = each.find('(')
# 			lat = each.find(')')
# 			a = each[pre:lat+1]
# 			t = each[pre+1:lat].split(',')[1].strip()
# 			new = each.replace(each[pre:lat+1], t).strip()+'\n'
# 			f_2.write(new)
# 			print("**:", new)


def draw_l_result():
	with open(read_file_path, "r") as f:
		t_100_x = []
		t_200_x = []
		t_300_x = []
		t_400_x = []
		t_500_x = []
		t_600_x = []
		t_700_x = []
		t_100_y = []
		t_200_y = []
		t_300_y = []
		t_400_y = []
		t_500_y = []
		t_600_y = []
		t_700_y = []
		for each in f.readlines():
			each = each.strip()
			_alpha = each.split(',')[0]
			_t = each.split(',')[1]
			_l = each.split(',')[2]
			_result = Decimal(each.split(',')[3]).quantize(Decimal("0.00"))
			_time = Decimal(each.split(',')[4]).quantize(Decimal("0.00"))
			if _alpha == '0.99':
				if _t == '100':
					t_100_x.append(_l)
					t_100_y.append(_result)
				if _t == '200':
					t_200_x.append(_l)
					t_200_y.append(_result)
				if _t == '300':
					t_300_x.append(_l)
					t_300_y.append(_result)
				if _t == '400':
					t_400_x.append(_l)
					t_400_y.append(_result)
				if _t == '500':
					t_500_x.append(_l)
					t_500_y.append(_result)
				if _t == '600':
					t_600_x.append(_l)
					t_600_y.append(_result)
				if _t == '700':
					print(_result)
					t_700_x.append(_l)
					t_700_y.append(_result)
		print(t_700_x)
		print(t_700_y)
		plt.plot(t_100_x, t_100_y, label='alpha = 0.99, t = 100')
		plt.plot(t_200_x, t_200_y, label='alpha = 0.99, t = 200')
		plt.plot(t_300_x, t_300_y, label='alpha = 0.99, t = 300')
		plt.plot(t_400_x, t_400_y, label='alpha = 0.99, t = 400')
		plt.plot(t_500_x, t_500_y, label='alpha = 0.99, t = 500')
		plt.plot(t_600_x, t_600_y, label='alpha = 0.99, t = 600')
		plt.plot(t_700_x, t_700_y, label='alpha = 0.99, t = 700')
		# plt.axhline(y=30628, color='black', linestyle='-', label='benchmark')
		plt.legend()
		plt.ylabel("result")
		plt.xlabel("l")
		plt.show()



def draw_t_result():
	with open(read_file_path, "r") as f:
		l_10_x = []
		l_50_x = []
		l_100_x = []
		l_500_x = []
		l_1000_x = []
		l_2000_x = []
		l_5000_x = []
		l_10_y = []
		l_50_y = []
		l_100_y = []
		l_500_y = []
		l_1000_y = []
		l_2000_y = []
		l_5000_y = []
		for each in f.readlines():
			each = each.strip()
			_alpha = each.split(',')[0]
			_t = each.split(',')[1]
			_l = each.split(',')[2]
			_result = Decimal(each.split(',')[3]).quantize(Decimal("0.00"))
			_time = Decimal(each.split(',')[4]).quantize(Decimal("0.00"))
			if _alpha == '0.99':
				if _l == '10':
					l_10_x.append(_t)
					l_10_y.append(_result)
				if _l == '50':
					l_50_x.append(_t)
					l_50_y.append(_result)
				if _l == '100':
					l_100_x.append(_t)
					l_100_y.append(_result)
				if _l == '500':
					l_500_x.append(_t)
					l_500_y.append(_result)
				if _l == '1000':
					l_1000_x.append(_t)
					l_1000_y.append(_result)
				if _l == '2000':
					l_2000_x.append(_t)
					l_2000_y.append(_result)
				if _l == '5000':
					print(_result)
					l_5000_x.append(_t)
					l_5000_y.append(_result)
		plt.plot(l_10_x, l_10_y, label='alpha = 0.99, l = 10')
		plt.plot(l_50_x, l_50_y, label='alpha = 0.99, l = 50')
		plt.plot(l_100_x, l_100_y, label='alpha = 0.99, l = 100')
		plt.plot(l_500_x, l_500_y, label='alpha = 0.99, l = 500')
		plt.plot(l_1000_x, l_1000_y, label='alpha = 0.99, l = 1000')
		plt.plot(l_2000_x, l_2000_y, label='alpha = 0.99, l = 2000')
		plt.plot(l_5000_x, l_5000_y, label='alpha = 0.99, l = 5000')
		# plt.axhline(y=30628, color='black', linestyle='-', label='benchmark')
		plt.legend()
		plt.ylabel("result")
		plt.xlabel("t")
		plt.show()

def draw_alpha_result():
	with open(read_file_path, "r") as f:
		t_100_x = []
		t_200_x = []
		t_300_x = []
		t_400_x = []
		t_500_x = []
		t_600_x = []
		t_700_x = []
		t_100_y = []
		t_200_y = []
		t_300_y = []
		t_400_y = []
		t_500_y = []
		t_600_y = []
		t_700_y = []
		for each in f.readlines():
			each = each.strip()
			_alpha = each.split(',')[0]
			_t = each.split(',')[1]
			_l = each.split(',')[2]
			_result = Decimal(each.split(',')[3]).quantize(Decimal("0.00"))
			_time = Decimal(each.split(',')[4]).quantize(Decimal("0.00"))
			if _l == '1000':
				if _t == '100':
					t_100_x.append(_alpha)
					t_100_y.append(_result)
				if _t == '200':
					t_200_x.append(_alpha)
					t_200_y.append(_result)
				if _t == '300':
					t_300_x.append(_alpha)
					t_300_y.append(_result)
				if _t == '400':
					t_400_x.append(_alpha)
					t_400_y.append(_result)
				if _t == '500':
					t_500_x.append(_alpha)
					t_500_y.append(_result)
				if _t == '600':
					t_600_x.append(_alpha)
					t_600_y.append(_result)
				if _t == '700':
					t_700_x.append(_alpha)
					t_700_y.append(_result)
		plt.plot(t_100_x, t_100_y, label='l = 1000, t = 100')
		plt.plot(t_200_x, t_200_y, label='l = 1000, t = 200')
		plt.plot(t_300_x, t_300_y, label='l = 1000, t = 300')
		plt.plot(t_400_x, t_400_y, label='l = 1000, t = 400')
		plt.plot(t_500_x, t_500_y, label='l = 1000, t = 500')
		plt.plot(t_600_x, t_600_y, label='l = 1000, t = 600')
		plt.plot(t_700_x, t_700_y, label='l = 1000, t = 700')
		# plt.axhline(y=30628, color='black', linestyle='-', label='benchmark')
		plt.legend()
		plt.ylabel("result")
		plt.xlabel("alpha")
		plt.show()


def draw_l_time():
	with open(read_file_path, "r") as f:
		t_100_x = []
		t_200_x = []
		t_300_x = []
		t_400_x = []
		t_500_x = []
		t_600_x = []
		t_700_x = []
		t_100_y = []
		t_200_y = []
		t_300_y = []
		t_400_y = []
		t_500_y = []
		t_600_y = []
		t_700_y = []
		for each in f.readlines():
			each = each.strip()
			_alpha = each.split(',')[0]
			_t = each.split(',')[1]
			_l = each.split(',')[2]
			_result = Decimal(each.split(',')[3]).quantize(Decimal("0.00"))
			_time = Decimal(each.split(',')[4]).quantize(Decimal("0.00"))
			if _alpha == '0.99':
				if _t == '100':
					t_100_x.append(_l)
					t_100_y.append(_time)
				if _t == '200':
					t_200_x.append(_l)
					t_200_y.append(_time)
				if _t == '300':
					t_300_x.append(_l)
					t_300_y.append(_time)
				if _t == '400':
					t_400_x.append(_l)
					t_400_y.append(_time)
				if _t == '500':
					t_500_x.append(_l)
					t_500_y.append(_time)
				if _t == '600':
					t_600_x.append(_l)
					t_600_y.append(_time)
				if _t == '700':
					print(_result)
					t_700_x.append(_l)
					t_700_y.append(_time)
		print(t_700_x)
		print(t_700_y)
		plt.plot(t_100_x, t_100_y, label='alpha = 0.99, t = 100')
		plt.plot(t_200_x, t_200_y, label='alpha = 0.99, t = 200')
		plt.plot(t_300_x, t_300_y, label='alpha = 0.99, t = 300')
		plt.plot(t_400_x, t_400_y, label='alpha = 0.99, t = 400')
		plt.plot(t_500_x, t_500_y, label='alpha = 0.99, t = 500')
		plt.plot(t_600_x, t_600_y, label='alpha = 0.99, t = 600')
		plt.plot(t_700_x, t_700_y, label='alpha = 0.99, t = 700')
		# plt.axhline(y=30628, color='black', linestyle='-', label='benchmark')
		plt.legend()
		plt.ylabel("time(s)")
		plt.xlabel("l")
		plt.show()


def draw_t_time():
	with open(read_file_path, "r") as f:
		l_10_x = []
		l_50_x = []
		l_100_x = []
		l_500_x = []
		l_1000_x = []
		l_2000_x = []
		l_5000_x = []
		l_10_y = []
		l_50_y = []
		l_100_y = []
		l_500_y = []
		l_1000_y = []
		l_2000_y = []
		l_5000_y = []
		for each in f.readlines():
			each = each.strip()
			_alpha = each.split(',')[0]
			_t = each.split(',')[1]
			_l = each.split(',')[2]
			_result = Decimal(each.split(',')[3]).quantize(Decimal("0.00"))
			_time = Decimal(each.split(',')[4]).quantize(Decimal("0.00"))
			if _alpha == '0.99':
				if _l == '10':
					l_10_x.append(_t)
					l_10_y.append(_time)
				if _l == '50':
					l_50_x.append(_t)
					l_50_y.append(_time)
				if _l == '100':
					l_100_x.append(_t)
					l_100_y.append(_time)
				if _l == '500':
					l_500_x.append(_t)
					l_500_y.append(_time)
				if _l == '1000':
					l_1000_x.append(_t)
					l_1000_y.append(_time)
				if _l == '2000':
					l_2000_x.append(_t)
					l_2000_y.append(_time)
				if _l == '5000':
					print(_result)
					l_5000_x.append(_t)
					l_5000_y.append(_time)
		plt.plot(l_10_x, l_10_y, label='alpha = 0.99, l = 10')
		plt.plot(l_50_x, l_50_y, label='alpha = 0.99, l = 50')
		plt.plot(l_100_x, l_100_y, label='alpha = 0.99, l = 100')
		plt.plot(l_500_x, l_500_y, label='alpha = 0.99, l = 500')
		plt.plot(l_1000_x, l_1000_y, label='alpha = 0.99, l = 1000')
		plt.plot(l_2000_x, l_2000_y, label='alpha = 0.99, l = 2000')
		plt.plot(l_5000_x, l_5000_y, label='alpha = 0.99, l = 5000')
		# plt.axhline(y=30628, color='black', linestyle='-', label='benchmark')
		plt.legend()
		plt.ylabel("time(s)")
		plt.xlabel("t")
		plt.show()


def draw_alpha_time():
	with open(read_file_path, "r") as f:
		t_100_x = []
		t_200_x = []
		t_300_x = []
		t_400_x = []
		t_500_x = []
		t_600_x = []
		t_700_x = []
		t_100_y = []
		t_200_y = []
		t_300_y = []
		t_400_y = []
		t_500_y = []
		t_600_y = []
		t_700_y = []
		for each in f.readlines():
			each = each.strip()
			_alpha = each.split(',')[0]
			_t = each.split(',')[1]
			_l = each.split(',')[2]
			_result = Decimal(each.split(',')[3]).quantize(Decimal("0.00"))
			_time = Decimal(each.split(',')[4]).quantize(Decimal("0.00"))
			if _l == '1000':
				if _t == '100':
					t_100_x.append(_alpha)
					t_100_y.append(_time)
				if _t == '200':
					t_200_x.append(_alpha)
					t_200_y.append(_time)
				if _t == '300':
					t_300_x.append(_alpha)
					t_300_y.append(_time)
				if _t == '400':
					t_400_x.append(_alpha)
					t_400_y.append(_time)
				if _t == '500':
					t_500_x.append(_alpha)
					t_500_y.append(_time)
				if _t == '600':
					t_600_x.append(_alpha)
					t_600_y.append(_time)
				if _t == '700':
					t_700_x.append(_alpha)
					t_700_y.append(_time)
		plt.plot(t_100_x, t_100_y, label='l = 1000, t = 100')
		plt.plot(t_200_x, t_200_y, label='l = 1000, t = 200')
		plt.plot(t_300_x, t_300_y, label='l = 1000, t = 300')
		plt.plot(t_400_x, t_400_y, label='l = 1000, t = 400')
		plt.plot(t_500_x, t_500_y, label='l = 1000, t = 500')
		plt.plot(t_600_x, t_600_y, label='l = 1000, t = 600')
		plt.plot(t_700_x, t_700_y, label='l = 1000, t = 700')
		# plt.axhline(y=30628, color='black', linestyle='-', label='benchmark')
		plt.legend()
		plt.ylabel("time(s)")
		plt.xlabel("alpha")
		plt.show()


# draw_t_result()
# draw_t_time()
# draw_alpha_result()
draw_alpha_time()
# draw_l_result()
# draw_l_time()