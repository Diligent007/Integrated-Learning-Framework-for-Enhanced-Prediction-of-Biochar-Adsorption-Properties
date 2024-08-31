import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# 加载数据
# data = pd.read_excel('Inputs(1).xlsx')  # 你的输入数据文件路径
# labels = pd.read_excel('Outputs-O(1).xlsx')  # 你的输出数据文件路径
x_data = pd.read_excel("YB-Inputs.xlsx")
y_data = pd.read_excel("YB-Outputs.xlsx")
x_c = x_data.columns
x_c = pd.DataFrame(x_c).T
y_c = y_data.columns
y_c = pd.DataFrame(y_c).T
x_data.columns = list(range(len(x_data.columns)))
y_data.columns = [0]

x_data = pd.concat([x_c, x_data], axis=0)
y_data = pd.concat([y_c, y_data], axis=0)
XG_param_space = {
    'n_estimators': (10, 200),
    'max_depth': (1, 20),
    'learning_rate': (0.01, 0.5),
    'min_child_weight': (1, 20),
    'subsample': (0.1, 1),
    'colsample_bytree': (0.1, 1)
}

RF_param_space = {
    'n_estimators': (10, 200),
    'max_depth': (1, 20),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 20)
}

RF_optimal_params = BayesSearchCV(RandomForestRegressor(), RF_param_space, n_iter=50, random_state=42, n_jobs=-1, cv=5,
                                  scoring="neg_root_mean_squared_error")
RF_optimal_params.fit(x_data, y_data)
print()
# 获取最优参数
best_RF_params = RF_optimal_params.best_params_
print(best_RF_params)
XG_optimal_params = BayesSearchCV(XGBRegressor(), XG_param_space, n_iter=50, random_state=42, n_jobs=-1, cv=5,
                                  scoring="neg_root_mean_squared_error")
XG_optimal_params.fit(x_data, y_data)
# 获取最优参数
best_XG_params = XG_optimal_params.best_params_
print(best_XG_params)
RF_model = RandomForestRegressor()
XG_model = XGBRegressor()
from sklearn.ensemble import VotingRegressor

ensemble_bytePair = VotingRegressor(estimators=[('RF', RF_model), ('XG', XG_model)])
# Define a range of weights
weights = np.linspace(0, 1, 50)
weight_combinations = [(w, 1 - w) for w in weights]

param_grid = {'weights': weight_combinations}

scorer = make_scorer(r2_score)

grid_search = GridSearchCV(estimator=ensemble_bytePair, param_grid=param_grid, scoring=scorer, cv=5, verbose=3)
grid_search.fit(x_data, y_data)
best_weights = grid_search.best_params_['weights']
print(f"网格搜索 Best Weights: {best_weights}")

ensemble_bytePair = VotingRegressor(estimators=[('RF', RF_model), ('XG', XG_model)],
                                    weights=best_weights)
# 使用 Shap 计算解释值
ensemble_bytePair.fit(x_data,y_data)
# explainer = shap.TreeExplainer(ensemble_bytePair)
explainer = shap.KernelExplainer(ensemble_bytePair.predict, shap.sample(x_data, 100))

shap_values = explainer.shap_values(x_data)



columns = ["C", "H", "N", "O", "S",
           "Ash", "VM", "FC", "Time", "Temperature", "Reaction rate", "PH"]



# 将 SHAP values 转换为 DataFrame
shap_df = pd.DataFrame(shap_values, columns=columns)
shap_df *= 0.01
# Save to a CSV file
shap_df.to_csv('shap_values.csv', index=False)

# Set the font to Times New Roman for all text in the plot
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15  # Set the base font size

# Create a figure object and set the figure size
plt.figure(figsize=(15, 10))
# Generate SHAP summary plot and multiply SHAP values by 100
shap_values *= 100
shap.summary_plot(shap_values, x_data, feature_names=columns, show=False)

# Set the xlabel and ylabel using updated font size
plt.xlabel('Yield+BET SHAP value (impact on model output)')
plt.ylabel('Features')

# Adjust the tick parameters for both axes using updated font size
plt.tick_params(axis='both', which='major', labelsize=15)

# If a legend is needed (summary_plot usually does not have a legend)
# plt.legend(prop={'size': 15, 'family': 'Times New Roman'})

# Save and show the plot
plt.savefig('C-SHAP-2.png', bbox_inches='tight')  # Save the figure with tight bounding box
plt.show()