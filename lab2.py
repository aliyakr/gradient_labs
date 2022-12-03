import numpy as np 

def function(x, y):
	f = (x**2 + y**2 - 1)**2 - (x + y - 1)**2

	return f 

def calculateGradients(x, y, delta = 0.001):
	gradientX = (function(x + delta, y) - function(x, y)) / delta
	gradientY = (function(x, y + delta) - function(x, y)) / delta 

	return (gradientX, gradientY)

def main():
	x = float(input("Введите начальное значение для x: "))
	y = float(input("Введите начальное значение для y: "))
	eps1 = float(input("Введите eps1: "))
	eps2 = float(input("Введите eps2: "))
	M = int(input("Введите предельное число итераций: "))
	t = float(input("Задайте величину шага t: "))

	k = 0
	while True:
		print(x, y)

		gradients = calculateGradients(x, y)
		gradientNorm = np.linalg.norm(gradients)

		if gradientNorm < eps1:
			break
		else:
			if k >= M:
				break 
			else:
				while True:
					lastPoints = (x, y)

					gradients = calculateGradients(x, y)

					gradientX = gradients[0]
					gradientY = gradients[1]

					x = x - t * gradientX
					y = y - t * gradientY

					if (function(x, y) - function(lastPoints[0], lastPoints[1])) < 0:
						break
					else:
						t = t / 2

				normOfPoints = np.linalg.norm((x - lastPoints[0], y - lastPoints[1]))

				if normOfPoints < eps2 and np.abs((function(x, y) - function(lastPoints[0], lastPoints[1]))) < eps2:
					break

	print(x, y)


if __name__ == "__main__":
	main()
else:
	print("Файл является не подключаемым, а исполняемым!")
