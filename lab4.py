import numpy as np 

def function(x, y):
	f = (x**2 + y**2 - 1)**2 - (x + y - 1)**2

	return f

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

def calculateHessian(x, y, delta = 0.001):
	hessianX = (function(x + delta, y) - 2 * function(x, y) + function(x - delta, y)) / delta ** 2
	hessianY = (function(x, y + delta) - 2 * function(x, y) + function(x, y - delta)) / delta ** 2
	hessianXY = (function(x + delta, y + delta) - function(x, y + delta) - function(x + delta, y) + function(x, y)) / delta ** 2

	hessian = np.array([[hessianX, hessianXY],
						[hessianXY, hessianY]])

	return hessian


def main():
	x = float(input("Введите начальное значение для x: "))
	y = float(input("Введите начальное значение для y: "))
	eps1 = float(input("Введите первую погрешность eps1: "))
	eps2 = float(input("Введите вторую погрешность eps2: "))
	M = int(input("Введите максимальное число итераций: "))

	k = 0
	nextA = np.eye(2)
	points = np.array((x, y))
	lastPoints = points

	while True:
		gradientsOfNextPoints = np.transpose(calculateGradients(points[0], points[1]))
		gradientOfLastPoints = calculateGradients(lastPoints[0], lastPoints[1])

		normOfGradient = np.linalg.norm(gradientsOfNextPoints)

		if normOfGradient < eps1:
			break
		else:
			if k >= M:
				break
			else:
				if k > 0:
					deltaG = gradientsOfNextPoints - gradientOfLastPoints
					deltaPoints = points - lastPoints

					deltaG = deltaG.reshape(-1, 1)
					deltaPoints = deltaPoints.reshape(-1, 1)

					lastA = (deltaPoints @ np.transpose(deltaPoints)) / (np.transpose(deltaPoints) @ deltaG) - (nextA @ deltaG @ np.transpose(deltaG) @ nextA) / (np.transpose(deltaG) @ nextA @ deltaG)
					nextA = nextA + lastA


				d = -1 * nextA @ gradientsOfNextPoints

				t = calculateStep(points[0], points[1], d)[0, 0]

				lastPoints = points 
				points = points - t * d

				normOfLastPoints = np.linalg.norm((points[0] - lastPoints[0], points[1] - lastPoints[1]))

				if normOfLastPoints < eps1 and (function(points[0], points[1]) - function(lastPoints[0], lastPoints[1])) < eps2:
					break
				else:
					k += 1

				print(f"Итерация: {k}. Точки: {points}. Шаг: {t}")
				# print(f"Итерация: {k}. Точки: {points}")

	print(f"Точки: {points}")










if __name__ == "__main__":
	main()
else:
	print("0")
