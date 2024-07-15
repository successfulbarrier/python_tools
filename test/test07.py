import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 生成示例数据
X = np.linspace(0, 10, 10)
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

# 创建样条插值
X_smooth = np.linspace(X.min(), X.max(), 300)
spl = make_interp_spline(X, y, k=3)  # k=3表示三次样条
y_smooth = spl(X_smooth)

# 可视化
plt.scatter(X, y, color='blue')
plt.plot(X_smooth, y_smooth, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Spline Interpolation')
plt.show()