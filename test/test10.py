import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

# 定义EMA函数
def exponential_moving_average(data, alpha):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema

# 计算平滑数据
alpha = 0.1  # 平滑因子
y_smooth = exponential_moving_average(y, alpha)

# 可视化
plt.plot(X, y, color='blue', label='Original Data')
plt.plot(X, y_smooth, color='red', label='Smoothed Data (EMA)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Exponential Moving Average')
plt.legend()
plt.show()