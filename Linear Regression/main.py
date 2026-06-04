import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('dataset.csv')


def loss_function(m, b , points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i, 0]
        y = points.iloc[i, 1]
        total_error += (y - (m*x + b))**2

    return total_error / float(len(points))


def gradient_descent(m_now,b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i, 0]
        y = points.iloc[i, 1]

        m_gradient += (-2/n) * x * (y - (m_now*x + b_now))
        b_gradient += (-2/n) * (y - (m_now*x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b


m = 0
b = 0
L = 0.001
epochs = 300

for i in range(epochs):
    m, b = gradient_descent(m,b,data,L)
    if i % 50 == 0:
        print(loss_function(m, b, data))

print("m: ",m)
print("b: ",b)
print("L: ",L)

x = data.iloc[:,0]
y_pred = m * x + b

plt.scatter(x, data.iloc[:,1])
plt.plot(x, y_pred)
plt.show()