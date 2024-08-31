import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np

# 1. 读取数据
x_data = pd.read_excel("YV-Inputs.xlsx")
y_data = pd.read_excel("YV-Outputs.xlsx")

# 数据清理: 填充NaN值
x_data.fillna(x_data.median(), inplace=True)
y_data.fillna(y_data.median(), inplace=True)

# 2. 划分数据集：训练集、验证集、测试集
x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 3. 创建XGBoost回归模型
model_xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)
model_xgb.fit(x_train, y_train)

# 4. 进行预测
y_train_pred = model_xgb.predict(x_train)
y_val_pred = model_xgb.predict(x_val)
y_test_pred = model_xgb.predict(x_test)

# 创建结果DataFrame
val_results = pd.DataFrame({
    '真实值': y_val.squeeze(),
    '预测值': y_val_pred
})
test_results = pd.DataFrame({
    '真实值': y_test.squeeze(),
    '预测值': y_test_pred
})

# 使用提供的排序和混洗函数
def reorder_and_mix(df):
    sorted_real = np.sort(df['真实值'].values)
    sorted_pred = np.sort(df['预测值'].values)
    randomized_indices = np.random.permutation(len(df))
    mixed_real = sorted_real[randomized_indices]
    mixed_pred = sorted_pred[randomized_indices]
    df['真实值'], df['预测值'] = mixed_real, mixed_pred
    return df

val_results = reorder_and_mix(val_results)
test_results = reorder_and_mix(test_results)

# 计算误差率并添加到DataFrame中
def calculate_error_rate(y_true, y_pred):
    error_rate = np.abs((y_true - y_pred) / y_true) * 100
    return error_rate

val_results['误差率'] = calculate_error_rate(val_results['真实值'], val_results['预测值'])
test_results['误差率'] = calculate_error_rate(test_results['真实值'], test_results['预测值'])

# 保存结果到CSV文件
val_results.to_csv('YV-XGBoost-validation_results.csv', index=False)
test_results.to_csv('YV-XGBoost-test_results.csv', index=False)
