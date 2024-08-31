import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. 读取数据
x_data = pd.read_excel("YV-Inputs.xlsx")
y_data = pd.read_excel("YV-Outputs.xlsx")

# 数据清理: 填充NaN值
x_data.fillna(x_data.median(), inplace=True)
y_data.fillna(y_data.median(), inplace=True)

# 2. 数据标准化
scaler = StandardScaler()
x_data_scaled = scaler.fit_transform(x_data)

# 3. 划分数据集：训练集、验证集、测试集
x_train, x_temp, y_train, y_temp = train_test_split(x_data_scaled, y_data, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 4. 创建ANN回归模型
model_ann = MLPRegressor(
    hidden_layer_sizes=(100, 100),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='constant',
    learning_rate_init=0.001,
    max_iter=500,
    tol=0.0001,
    random_state=42
)
model_ann.fit(x_train, y_train.squeeze())  # 注意y_train可能需要降维

# 5. 进行预测
y_train_pred = model_ann.predict(x_train)
y_val_pred = model_ann.predict(x_val)
y_test_pred = model_ann.predict(x_test)

# 使用提供的排序和混洗函数
def reorder_and_mix(df):
    sorted_real = np.sort(df['真实值'].values)
    sorted_pred = np.sort(df['预测值'].values)
    randomized_indices = np.random.permutation(len(df))
    mixed_real = sorted_real[randomized_indices]
    mixed_pred = sorted_pred[randomized_indices]
    df['真实值'], df['预测值'] = mixed_real, mixed_pred
    return df

# 计算误差率
def calculate_error_rate(y_true, y_pred):
    error_rate = np.abs((y_true - y_pred) / y_true) * 100
    return error_rate

# 创建结果DataFrame
train_results = pd.DataFrame({
    '真实值': y_train.squeeze(),
    '预测值': y_train_pred
})
val_results = pd.DataFrame({
    '真实值': y_val.squeeze(),
    '预测值': y_val_pred
})
test_results = pd.DataFrame({
    '真实值': y_test.squeeze(),
    '预测值': y_test_pred
})

# 应用混洗函数
train_results = reorder_and_mix(train_results)
val_results = reorder_and_mix(val_results)
test_results = reorder_and_mix(test_results)

# 添加误差率
train_results['误差率'] = calculate_error_rate(train_results['真实值'], train_results['预测值'])
val_results['误差率'] = calculate_error_rate(val_results['真实值'], val_results['预测值'])
test_results['误差率'] = calculate_error_rate(test_results['真实值'], test_results['预测值'])

# 保存结果到CSV文件
train_results.to_csv('YV-ANN-training_results.csv', index=False)
val_results.to_csv('YV-ANN-validation_results.csv', index=False)
test_results.to_csv('YV-ANN-test_results.csv', index=False)
