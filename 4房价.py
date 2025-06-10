import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
pd.set_option('display.max_rows', None)
# 加载数据
data = pd.read_excel('C:/Users/秦/Desktop/fangjia.xlsx')

# 查看数据前几行
print(data.head())

# 将房价设为目标变量y，其他列设为特征X
X = data.drop('房价（万元）', axis=1)
y = data['房价（万元）']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

#lasso回归
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

# 预测
y_pred_lasso = lasso_model.predict(X_test)

# 评估模型
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f'MSE (Lasso Regression): {mse_lasso}')
print(f'R² Score (Lasso Regression): {r2_lasso}')

#岭回归
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# 预测
y_pred_ridge = ridge_model.predict(X_test)

# 评估模型
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f'MSE (Ridge Regression): {mse_ridge}')
print(f'R² Score (Ridge Regression): {r2_ridge}')