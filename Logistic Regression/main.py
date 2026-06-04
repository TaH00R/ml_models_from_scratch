import numpy as np

def sigmoid(z):
    return 1.0/ (1.0 + np.exp(-z))


def gradient(theta, X, y):
    m = y.size #NO OF INSTANCES
    return (X.T @ (sigmoid(X@theta) - y))/m


def gradient_descent(X, y, alpha = 0.1, num_iter = 1000, tol = 1e-7):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(X_b.shape[1])

    for i in range(num_iter):
        grad = gradient(theta, X_b, y)
        theta -= alpha * grad

        if np.linalg.norm(grad) < tol:
            break

    return theta


def predict_proba(theta, X):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X_b@theta)


def predict(theta, X, threshold=0.5):
    return (predict_proba(theta, X)>=threshold).astype(int)


from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

theta_hat = gradient_descent(X_train_scaled, y_train)
y_pred_train = predict(theta_hat, X_train_scaled)
y_pred_test = predict(theta_hat, X_test_scaled)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(train_accuracy, test_accuracy)


