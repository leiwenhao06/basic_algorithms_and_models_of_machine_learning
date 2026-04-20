# 一元线性回归 (Simple Linear Regression) 详解

## 1. 什么是线性回归？

线性回归是利用数理统计中的回归分析，来确定两种或两种以上变量间相互依赖的**定量关系**的一种统计分析方法。

**一元线性回归**只涉及一个自变量 \( x \) 和一个因变量 \( y \)，并且它们之间的关系可以用一条直线近似表示：
\[
y \approx \beta_0 + \beta_1 x
\]
- **\( \beta_0 \)**：截距 (Intercept)
- **\( \beta_1 \)**：斜率 (Slope)，即 \( x \) 每变化一个单位，\( y \) 的平均变化量。

---

## 2. 核心思想：最小二乘法

我们的目标是找到一条“最好”的直线，使得所有样本点距离这条直线的**竖直距离的平方和**最小。

### 2.1 损失函数 (Loss Function)
定义残差 (Residual) 为真实值 \( y_i \) 与预测值 \( \hat{y}_i \) 之差：
\[
e_i = y_i - \hat{y}_i = y_i - (\beta_0 + \beta_1 x_i)
\]

我们使用**均方误差 (Mean Squared Error, MSE)** 作为损失函数：
\[
J(\beta_0, \beta_1) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2
\]

### 2.2 参数求解（闭式解 / 正规方程）

为了最小化 \( J \)，分别对 \( \beta_0 \) 和 \( \beta_1 \) 求偏导并令其为 0。

**对 \( \beta_0 \) 求偏导：**
\[
\frac{\partial J}{\partial \beta_0} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i) = 0
\]
解得：
\[
\beta_0 = \bar{y} - \beta_1 \bar{x}
\]
其中 \( \bar{x} \) 和 \( \bar{y} \) 分别是 \( x \) 和 \( y \) 的样本均值。

**对 \( \beta_1 \) 求偏导：**
\[
\frac{\partial J}{\partial \beta_1} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - \beta_0 - \beta_1 x_i) = 0
\]
代入 \( \beta_0 \) 表达式，整理后得到斜率 \( \beta_1 \) 的公式：
\[
\beta_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
\]
> 分子为 \( x \) 和 \( y \) 的协方差，分母为 \( x \) 的方差。

---

## 3. 模型评估指标

训练好模型后，我们需要量化它的好坏。

| 指标 | 公式 | 含义 |
| :--- | :--- | :--- |
| **均方误差 (MSE)** | \(\frac{1}{n}\sum(y_i - \hat{y}_i)^2\) | 预测值与真实值差异平方的均值，越小越好 |
| **均方根误差 (RMSE)** | \(\sqrt{MSE}\) | MSE 的平方根，量纲与 y 一致，更直观 |
| **平均绝对误差 (MAE)** | \(\frac{1}{n}\sum \lvert y_i - \hat{y}_i \rvert\) | 绝对误差均值，对异常值不那么敏感 |
| **决定系数 (\( R^2 \))** | \(1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}\) | 回归平方和占总平方和的比例，越接近 1 拟合越好 |

---

## 4. Python 代码实现

以下代码分别使用**公式推导法**（手动实现）和 **Scikit-learn**（工业级实现）演示一元线性回归。

```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    """
    一元线性回归（最小二乘法闭式解）
    """
    def __init__(self):
        self.beta_0 = None  # 截距
        self.beta_1 = None  # 斜率

    def fit(self, X, y):
        """
        训练模型
        X: 形状为 (n_samples, ) 或 (n_samples, 1)
        y: 形状为 (n_samples, )
        """
        X = np.array(X).flatten()
        y = np.array(y).flatten()
        
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # 根据公式计算斜率 beta_1
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        self.beta_1 = numerator / denominator
        
        # 计算截距 beta_0
        self.beta_0 = y_mean - self.beta_1 * x_mean
        
        return self

    def predict(self, X):
        X = np.array(X).flatten()
        return self.beta_0 + self.beta_1 * X

    def score(self, X, y):
        """计算 R^2 决定系数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# ------------------- 示例运行 -------------------
if __name__ == "__main__":
    # 生成模拟数据：y = 3 * x + 5 + 噪声
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y = 3 * X + 5 + np.random.randn(50) * 2  # 加高斯噪声

    # 1. 手动实现模型
    model = SimpleLinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    print("=== 手动实现结果 ===")
    print(f"拟合直线: y = {model.beta_0:.4f} + {model.beta_1:.4f} * x")
    print(f"R^2 得分: {model.score(X, y):.4f}")

    # 2. 与 Scikit-learn 对比（验证正确性）
    from sklearn.linear_model import LinearRegression
    sk_model = LinearRegression()
    sk_model.fit(X.reshape(-1, 1), y)
    print("\n=== Scikit-learn 结果 ===")
    print(f"拟合直线: y = {sk_model.intercept_:.4f} + {sk_model.coef_[0]:.4f} * x")
    print(f"R^2 得分: {sk_model.score(X.reshape(-1, 1), y):.4f}")

    # 3. 可视化
    plt.scatter(X, y, alpha=0.7, label='真实数据')
    plt.plot(X, y_pred, color='red', linewidth=2, label='拟合直线')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('一元线性回归拟合效果')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
```

---

## 5. 梯度下降法视角（拓展）

虽然一元线性回归有完美的闭式解，但在**多元线性回归**或大规模数据中，通常使用**梯度下降法**迭代求解。

**参数更新公式：**
\[
\begin{aligned}
\beta_0 &:= \beta_0 - \alpha \frac{\partial J}{\partial \beta_0} \\
\beta_1 &:= \beta_1 - \alpha \frac{\partial J}{\partial \beta_1}
\end{aligned}
\]
其中 \( \alpha \) 为学习率。

计算偏导：
\[
\frac{\partial J}{\partial \beta_0} = -\frac{2}{n} \sum (y_i - \beta_0 - \beta_1 x_i)
\]
\[
\frac{\partial J}{\partial \beta_1} = -\frac{2}{n} \sum x_i (y_i - \beta_0 - \beta_1 x_i)
\]

> 在仓库的 `optimization/gradient_descent.py` 中，可以对比闭式解与梯度下降法在一元数据上的收敛过程。

---

## 6. 前提假设与局限性

线性回归模型成立需满足以下**经典假设**（违反时结论可能不可靠）：
1. **线性关系**：\( y \) 与 \( x \) 呈线性关系。
2. **独立性**：样本间相互独立。
3. **同方差性**：误差项方差恒定。
4. **正态性**：误差项服从正态分布。

**局限性**：
- 对异常值敏感（因为 MSE 会放大残差的平方）。
- 无法拟合非线性关系（可通过多项式回归、样条回归解决）。

---

## 7. 一句话总结

> 一元线性回归通过最小化均方误差寻找最佳拟合直线，其闭式解为 \( \beta_1 = \frac{\text{Cov}(x,y)}{\text{Var}(x)} \)，\( \beta_0 = \bar{y} - \beta_1\bar{x} \)，是机器学习入门的第一个基石模型。

---
