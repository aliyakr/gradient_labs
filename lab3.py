import numpy as np 

def checkSylvesterСriterion(invHessian):
  firstMinor = invHessian[0, 0]
  secondMinor = np.linalg.det(invHessian)

  # print(firstMinor, secondMinor)

  if firstMinor > 0 and secondMinor > 0:
    return True
  else:
    return False

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

def function(x, y):
  f = (x**2 + y**2 - 1)**2 - (x + y - 1)**2
  

  return f

def calculateStep(x, y, d):
  d = d.reshape(-1, 1)
  gradients = calculateGradients(x, y)
  hessian = calculateHessian(x, y)

  step = np.dot(gradients, d) / (np.dot(np.dot(np.transpose(d), hessian), d))

  return step

def main():
  x = float(input("Введите начальное значение для x: "))
  y = float(input("Введите начальное значение для y: "))
  eps1 = float(input("Введите значение для eps1: "))
  eps2 = float(input("Введите значение для eps2: "))
  M = int(input("Введите число итераций: "))

  k = 0

  while k <= M:
    lastPoints = (x, y)

    gradients = np.transpose(calculateGradients(x, y))
    normOfGradient = np.linalg.norm(gradients)

    if normOfGradient < eps1: 
      break 
    else:
      if k >= M:
        break
      else:
        hessian = calculateHessian(x, y)
        invHessian = np.linalg.inv(hessian)

        if checkSylvesterСriterion(invHessian):
          d = -1 * np.dot(invHessian, gradients)
          t = 1
        else:
          d = -1 * gradients
          t = calculateStep(x, y, d)[0, 0]

        x = x + t * d[0]
        y = y + t * d[1]

        normOfLastPoints = np.linalg.norm((x - lastPoints[0], y - lastPoints[1]))

        if normOfLastPoints < eps1 and (function(x, y) - function(lastPoints[0], lastPoints[1])) < eps2:
          break
        else:
          k += 1

    # print(f"Точки: {x, y}. Градиент: {gradients}. Гессиан: {hessian}. Обратный гессиан: {invHessian}")
    print(f"Итерация: {k}. Точки: {x, y}. Шаг: {t}")




if __name__ == "__main__":
  main()
else:
  print("0")
