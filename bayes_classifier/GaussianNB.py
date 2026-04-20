# 使用朴素贝叶斯分类器对乳腺癌数据集建立分类模型并评估预测性能

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB          # 对应连续型数据导入分类模型
from sklearn.metrics import accuracy_score          
from sklearn.datasets import load_breast_cancer     # 导入数据集
import pandas as pd

# 加载数据集
data = load_breast_cancer()
X = data['data']
y = data['target']      # 分类标签
feature_names = data['feature_names']   # 特征名称
target_names = data['target_names']     # 分类标签名称

# 设置比例将数据分成训练集和测试集，采用分层分割模式，比例为7:3，设置随机种子保证结果可复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 训练分类模型
model = GaussianNB()
model.fit(X_train, y_train)

# 使用分类模型对测试集样本进行预测
y_pred_test = model.predict(X_test)

# 模型性能评估，使用的指标为准确率
acc_test = accuracy_score(y_test, y_pred_test)      # 模型在测试集上的表现
print(f"模型在测试集上的准确率为：{acc_test:.4f}")

y_pred_train = model.predict(X_train)
acc_train = accuracy_score(y_train, y_pred_train)   # 模型在训练集上的表现
print(f"模型在训练集上的准确率为：{acc_train:.4f}")

# 观察每个样本属于不同类别的概率
y_prob = model.predict_proba(X_test)
prob_df = pd.DataFrame(y_prob, columns=model.classes_)
print(y_prob)

# 本案例基于 sklearn 提供的乳腺癌数据集（load_breast_cancer），构建朴素贝叶斯分类模型完成二分类任务。整体求解流程如下：
# 1. 数据处理与划分
# 数据集包含 569 个样本，每个样本包含 30 个连续特征（如肿瘤半径、纹理等）。分类标签为二分类，0表示恶性（malignant）1表示良性（benign）。采用 train_test_split 对数据进行划分，训练集与测试集划分比例为 = 7 : 3，使用 stratify=y 保证类别分布一致，设置 random_state=42 保证实验可复现。
# 2. 模型选择
# 选用高斯朴素贝叶斯（Gaussian Naive Bayes），原因是数据特征为连续型变量且高斯分布假设符合该类医学数据的统计特性。核心假设为各特征条件独立且每个特征在类别条件下服从高斯分布。
# 3. 模型训练与预测
# 使用训练集调用 model.fit(X_train, y_train) 进行参数学习，在训练集评估拟合能力，在测试集评估泛化能力，分别进行预测 y_pred_train = model.predict(X_train), y_pred_test = model.predict(X_test)。
# 4. 性能评估方法
# 采用准确率（Accuracy） 作为评价指标，Accuracy=正确分类样本数/总样本数。同时输出训练集准确率与测试集准确率。
# 5. 概率输出分析
# 使用 predict_proba() 获取每个样本属于各类别的概率分布，y_prob = model.predict_proba(X_test)，可用于分类置信度分析与后续阈值调节或ROC分析。
