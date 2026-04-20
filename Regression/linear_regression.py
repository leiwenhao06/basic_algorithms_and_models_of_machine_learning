# 使用一元线性回归求解房屋面积与交易价格之间的关系
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据集
X = np.array([100, 120, 150, 200, 220, 250, 300, 350, 400, 450, 500, 600])        # 房屋面积
y = np.array([570, 610, 645, 745, 798, 845, 942, 1145, 1370, 1545, 1730, 1880])   # 交易价格

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# 创建线性回归对象
model = LinearRegression()

# 训练模型
model.fit(X, y)

# 存储参数
w0 = model.intercept_[0]
w1 = model.coef_[0, 0]

# 显示具体一元线性回归公式
print(f"得到的回归模型为：{w1:.4f} * area + {w0:.4f}")

#使用回归模型预测
y_pred = model.predict(X)

# 计算均方根误差与决定系数
mse_root = mean_squared_error(y, y_pred) ** 0.5
print(f"均方根误差为：{mse_root:.4f}")
rsquare = r2_score(y, y_pred)
print(f"决定系数：{rsquare:.4f}")

# 绘制散点图与回归线
plt.rcParams['font.sans-serif'] = ['SimHei']  #显示中文标签
plt.scatter(X, y, color="black", label='训练样本点')
plt.plot(X, y_pred, c="blue", linewidth=1, label='回归线')
plt.legend(loc='upper left', fontsize=12)
plt.xlabel('房屋面积', fontsize=12)
plt.ylabel('交易价格', fontsize=12)
plt.show()
