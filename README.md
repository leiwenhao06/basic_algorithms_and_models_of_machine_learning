# Basic Algorithms and Models of Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

本仓库是我在学习机器学习过程中的**笔记总结**，包含了使用机器学习方法求解回归、分类与聚类问题的经典案例，涵盖所有的基础模型与算法，给出了基于Python的代码实现并附有对应知识的学习笔记。

目标是汇总核心算法与模型，以此深入理解数学原理与工程细节。所有实现均配有详细的 Markdown 笔记，适合初学者相互对照学习。

---

## 📁 仓库结构

```text
basic_algorithms_and_models_of_machine_learning/
│
├── supervised_learning/         # 监督学习
│   ├── regression/              # 回归算法
│   │   ├── linear_regression.py
│   │   └── 线性回归笔记.md
|   |   ├── multivariate_linear_regression.py
|   |   └── 多元线性回归笔记.md
│   │   ├── polynomial_regression.py
│   │   └── 多项式回归笔记.md
│   │
│   └── classification/          # 分类算法
│       ├── logistic_regression.py
│       └── 逻辑回归笔记.md
│       ├── knn.py
│       └── K近邻笔记.md
│       ├── decision_tree.py
│       └── 决策树笔记.md
│
├── unsupervised_learning/       # 无监督学习
│   ├── clustering/              # 聚类算法
│   │   ├── kmeans.py
│   │   └── K均值聚类笔记.md
│   │   ├── hierarchical_clustering.py
│   │   └── 层次聚类笔记.md
│   │
│   └── dimensionality_reduction/# 降维算法
│       ├── pca.py
│       └── 主成分分析笔记.md
│
├── neural_networks/             # 神经网络基础
│   ├── perceptron.py
│   ├── mlp_numpy.py             # 纯 NumPy 实现的多层感知机
│   └── backpropagation_notes.md # 反向传播详细推导笔记
│
├── optimization/                # 优化算法
│   ├── gradient_descent.py
│   ├── sgd_momentum.py
│   └── 优化器笔记.md
│
├── utils/                       # 工具函数（数据加载、评估指标等）
│   ├── data_loader.py
│   ├── metrics.py
│   └── visualization.py
│
├── notes/                       # 通用理论学习笔记
│   ├── bias_variance_tradeoff.md
│   ├── regularization.md
│   └── evaluation_metrics.md
│
├── datasets/                    # 示例数据集（小型 CSV）
│   └── README.md
│
├── requirements.txt             # Python 依赖包列表
└── README.md                    # 本文件
```

---

## 🚀 快速开始

### 环境要求
- Python 3.8 或更高版本
- 推荐使用虚拟环境（`venv` 或 `conda`）

### 安装依赖
```bash
git clone https://github.com/leiwenhao06/basic_algorithms_and_models_of_machine_learning.git
cd basic_algorithms_and_models_of_machine_learning
pip install -r requirements.txt
```

### 运行示例
每个算法文件均包含 `if __name__ == "__main__":` 入口，可直接运行测试。

```bash
# 运行线性回归示例
python supervised_learning/regression/linear_regression.py

# 运行 K-Means 聚类示例
python unsupervised_learning/clustering/kmeans.py
```

---

## 📖 笔记说明

- 每个算法目录下均配有同名的 **`.md` 笔记文件**。
- 笔记内容包含：
  - 算法核心思想与数学推导
  - 案例问题求解思路
  - 关键参数影响分析
  - 优缺点及适用场景
- 建议**先看笔记理解原理，再阅读代码**，效果更佳。

---

## 🎯 实现清单与进度

| 算法 / 模型 | 代码 | 笔记 | 状态 |
| :--- | :---: | :---: | :---: |
| **线性回归** (Linear Regression) | 🚧 | 🚧 | 进行中 |
| **多项式回归** (Polynomial Regression) | 🚧 | 🚧 | 已完成 |
| **逻辑回归** (Logistic Regression) | 🚧 | 🚧 | 已完成 |
| **K近邻** (K-Nearest Neighbors) | 🚧 | 🚧 | 已完成 |
| **决策树** (Decision Tree) | 🚧 | 🚧 | 进行中 |
| **K-Means 聚类** | 🚧 | 🚧 | 已完成 |
| **主成分分析** (PCA) | 🚧 | 🚧 | 已完成 |
| **多层感知机** (MLP with NumPy) | 🚧 | 🚧 | 进行中 |
| **梯度下降优化器** | 🚧 | 🚧 | 已完成 |

> ✅ 已完成 &nbsp;&nbsp;&nbsp; 🚧 进行中 &nbsp;&nbsp;&nbsp; ⏳ 计划中

---

## 📚 参考资料

- 《机器学习》（周志华）
- 《统计学习方法》（李航）
- [CS229: Machine Learning (Stanford)](http://cs229.stanford.edu/)
- [Pattern Recognition and Machine Learning (Bishop)](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
- Scikit-learn 官方文档

---

## 📝 许可证

本项目采用 [MIT License](LICENSE) 开源协议。你可以自由地使用、修改和分发代码，但请保留原作者声明。

---

## 🤝 贡献与联系

如果你发现代码有误、笔记有歧义，或者有任何改进建议，欢迎提 **Issue** 或 **Pull Request**。

⭐ **如果这个仓库对你学习机器学习有帮助，欢迎点一个 Star 支持一下！** ⭐
```
3. 如果暂时没有 `LICENSE` 文件，可以先删除许可证部分，或者后续在仓库里添加一个。
