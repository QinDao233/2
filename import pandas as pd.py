import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 读取Excel文件
file_path = r'C:\Users\秦\Desktop\correlation_matrix.xlsx'
df = pd.read_excel(file_path)
df = df.rename(columns={'x1': '身高', 'x2': '坐高', 'x3': '胸围', 'x4': '手臂长', 'x5': '肋围', 'x6': '腰围'})

# 选择需要进行PCA分析的列
features = ['身高', '坐高', '胸围', '手臂长', '肋围', '腰围']
df_data = df[features]

# 创建PCA实例
pca = PCA(n_components=6)

# 对数据进行拟合和转换
principalComponents = pca.fit_transform(df_data)

# 创建一个新的DataFrame来存储主成分
principalDf = pd.DataFrame(data = principalComponents, 
                           columns = [f'PC{i+1}' for i in range(6)])
print(principalDf.head())

# 输出每个主成分的方差贡献率
print("\nExplained Variance Ratios (Feature Importances):")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"Component PC{i+1}: {ratio*100:.2f}%")

# 计算并输出累计解释方差比
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
print("\nCumulative Explained Variance Ratios:")
for i, ratio in enumerate(cumulative_explained_variance):
    print(f"Up to Component PC{i+1}: {ratio*100:.2f}%")

# 创建条形图
x = np.arange(1, 7)
y = pca.explained_variance_ratio_

fig, ax = plt.subplots()
ax.bar(x, y, width=0.8, color='blue')  # 使用matplotlib的bar函数绘制条形图
ax.set_title('Variance Explained by Each Principal Component')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance Ratio')
ax.set_xticks(x)  # 设置x轴标签
ax.set_xticklabels([f'PC{i}' for i in range(1, 7)], rotation=0)  # 添加主成分标签
plt.show()

# 创建线形图
plt.plot(np.arange(1, 7), cumulative_explained_variance)
plt.title('Cumulative Variance Explained by Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(np.arange(1, 7), [f'PC{i}' for i in range(1, 7)])  # 添加主成分标签
plt.show()

# 创建散点图
plt.figure(figsize=(10, 8))
for idx, row in principalDf.iterrows():  # 遍历每一个数据点
    plt.scatter(row.iloc[0], row.iloc[1], marker='o')
    plt.annotate(idx + 1, xy=(row.iloc[0]+0.01, row.iloc[1]-0.01))  # 在每个点附近添加标注，注意这里idx+1
plt.title('Scatter Plot of PCA')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()