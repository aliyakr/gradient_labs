import numpy as np

def calculateHessian(x, y, delta = 0.001):
	hessianX = (function(x + delta, y) - 2 * function(x, y) + function(x - delta, y)) / delta ** 2
	hessianY = (function(x, y + delta) - 2 * function(x, y) + function(x, y - delta)) / delta ** 2
	hessianXY = (function(x + delta, y + delta) - function(x, y + delta) - function(x + delta, y) + function(x, y)) / delta ** 2

	hessian = np.array([[hessianX, hessianXY],
						[hessianXY, hessianY]])

	return hessian

def calculateGradients(x, y, delta = 0.001):
	gradientX = (function(x + delta, y) - function(x, y)) / delta 
	gradientY = (function(x, y + delta) - function(x, y)) / delta 

	return np.array((gradientX, gradientY))

def calculateStep(x, y, d):
	d = d.reshape(-1, 1)
	gradients = calculateGradients(x, y)
	hessian = calculateHessian(x, y)

	step = np.dot(gradients, d) / (np.dot(np.dot(np.transpose(d), hessian), d))

	return step

def function(x, y):
	f = (x**2 + y**2 - 1)**2 - (x + y - 1)**2 


	return f

def main():
	print("Введите начальные точки: ")
	points = list(map(int, input().split()))
	eps = float(input("Введите погрешность: "))

	k = 0
	lastPoints = points

	while True:
		gradientsOfPoints = calculateGradients(points[0], points[1])
		gradientsOfLastPoints = calculateGradients(lastPoints[0], lastPoints[1])

		lastPoints = points

		normGradientOfPoints = np.linalg.norm(gradientsOfPoints)
		normGradientOfLastPoints = np.linalg.norm(gradientsOfLastPoints)

		if normGradientOfPoints  < eps:
			break
		else:
			if k == 0:
				d = gradientsOfPoints
			else:
				b = normGradientOfPoints ** 2 / normGradientOfLastPoints ** 2
				d = gradientsOfPoints + b * gradientsOfLastPoints

			t = calculateStep(points[0], points[1], d)[0, 0]


			points = lastPoints - t * d
			k += 1

		print(f"k: {k}. Points: {points}. Step: {t}")



if __name__ == "__main__":
	main()
else:
	print("Файл явялется исполняемым, а не подключаемым!")
