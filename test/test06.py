import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 生成示例数据
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])

# 创建多项式特征
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 创建线性回归模型并拟合多项式特征
model = LinearRegression()
model.fit(X_poly, y)

# 预测
y_pred = model.predict(X_poly)

# 可视化
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.show()