import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 生成示例数据
X = np.array([1, 2, 3, 4, 5])
y = np.array([1, 3, 2, 3, 5])

# 创建Savitzky-Golay滤波器
y_smooth = savgol_filter(y, window_length=3, polyorder=2)

# 可视化
plt.scatter(X, y, color='blue')
plt.plot(X, y_smooth, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Savitzky-Golay Filter')
plt.show()