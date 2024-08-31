# import joblib
# import numpy as np
# import pandas as pd
# from pyswarm import pso
#
# # 加载数据
# x_data = pd.read_csv('YB-Inputs.csv').to_numpy()
# y_data = pd.read_csv('YB-Outputs.csv').to_numpy()
# model = joblib.load("融合模型.pkl")
#
# # 初始化存储最佳结果和损失的列表
# best_x = []
# best_y = []
# losses = []
#
# def objective_function(x):
#     global best_x, best_y, losses
#     # 检查前五个参数的累积和是否小于等于100
#     if np.sum(x[:5]) > 100:
#         print("没有满足前五个输入参数的和小于等于100的约束条件。")
#         return 1e10  # 使用一个大数作为惩罚，以阻止使用这些参数组合
#
#     predicted_y = model.predict([x])[0]
#     loss = -predicted_y  # 取预测值的负数以实现最大化输出
#
#     # 存储最佳结果和损失
#     if not best_y or predicted_y > max(best_y, default=float('-inf')):
#         best_x = x.copy()  # 存储当前的参数值
#         best_y = [predicted_y]  # 更新最大的预测输出值
#
#     losses.append(-loss)  # 损失列表记录正的预测值
#     return loss
#
# # 设置PSO算法的参数
# n_particles = 10
# lb = [6.87,0.14,0.38,0.43,0.10,1.31,7.23,1.82,15.00,300.00,5.00,2.00]   # 各参数的下界
# ub = [76.23,7.97,7.77,68.96,2.32,89.41,75.03,60.29,360,900,35,11.37]   # 各参数的上界
#
# # 使用PSO算法寻找最优参数
# try:
#     x_optimal, f_opt = pso(objective_function, lb, ub, swarmsize=n_particles, maxiter=100)
#     print("最优参数:", x_optimal)
#     predicted_best_y = model.predict([x_optimal])[0]
#     print("对应的最大输出值:", predicted_best_y)
# except Exception as e:
#     print("优化过程中出现错误:", e)
#
# # 保存结果
# if best_x:  # 如果有有效的最佳解
#     df_best = pd.DataFrame([list(best_x) + [predicted_best_y]], columns=[f'Feature_{i+1}' for i in range(len(best_x))] + ['Predicted_Y'])
#     df_best.to_csv('PSO_best_xy-1.csv', index=False)
#     df_losses = pd.DataFrame({'Loss': losses})
#     df_losses.to_csv('PSO_losses-1.csv', index=False)




import joblib
import numpy as np
import pandas as pd
from pyswarm import pso

# 加载数据
x_data = pd.read_csv('YB-Inputs.csv').to_numpy()
y_data = pd.read_csv('YB-Outputs.csv').to_numpy()
model = joblib.load("融合模型.pkl")

# 初始化存储最佳结果和损失的列表
best_x = None
best_y = []
losses = []

def objective_function(x):
    global best_x, best_y, losses
    # 检查前五个参数的累积和是否小于等于100
    # if np.sum(x[:5]) > 100:
    #     print("没有满足前五个输入参数的和小于等于100的约束条件。")
    #     return 1e10  # 使用一个大数作为惩罚，以阻止使用这些参数组合

    predicted_y = model.predict([x])[0]
    loss = -predicted_y  # 取预测值的负数以实现最大化输出

    # 存储最佳结果和损失
    if not best_y or predicted_y > max(best_y, default=float('-inf')):
        best_x = x.copy()  # 存储当前的参数值
        best_y = [predicted_y]  # 更新最大的预测输出值

    losses.append(-loss)  # 损失列表记录正的预测值
    return loss

# 设置PSO算法的参数
n_particles = 100
lb = [6.87, 0.14, 0.38, 0.43, 0.10, 1.31, 7.23, 1.82, 15.00, 300.00, 5.00, 2.00]   # 各参数的下界
ub = [76.23, 7.97, 7.77, 68.96, 2.32, 89.41, 75.03, 60.29, 360, 900, 35, 11.37]   # 各参数的上界

# 使用PSO算法寻找最优参数
try:
    x_optimal, f_opt = pso(objective_function, lb, ub, swarmsize=n_particles, maxiter=100)
    print("最优参数:", x_optimal)
    predicted_best_y = model.predict([x_optimal])[0]
    print("对应的最大输出值:", predicted_best_y)
except Exception as e:
    print("优化过程中出现错误:", e)

# 保存结果
if best_x is not None:  # 检查best_x是否已被赋值
    df_best = pd.DataFrame([list(best_x) + [predicted_best_y]], columns=[f'Feature_{i+1}' for i in range(len(best_x))] + ['Predicted_Y'])
    df_best.to_csv('PSO_best_xy.csv', index=False)
    df_losses = pd.DataFrame({'Loss': losses})
    df_losses.to_csv('PSO_losses.csv', index=False)
else:
    print("未找到满足条件的最佳解。")
