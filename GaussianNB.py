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

