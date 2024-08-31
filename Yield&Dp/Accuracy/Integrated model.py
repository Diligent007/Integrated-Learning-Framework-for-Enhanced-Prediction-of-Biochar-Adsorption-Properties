import joblib
import pandas as pd
from scipy.optimize import minimize

from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

from sklearn.model_selection import train_test_split, GridSearchCV
from skopt import BayesSearchCV
from catboost import CatBoostRegressor
import numpy as np
import warnings

warnings.filterwarnings("ignore")
x_data = pd.read_excel("YD-Inputs.xlsx")
y_data = pd.read_excel("YD-Outputs.xlsx")
x_c = x_data.columns
x_c = pd.DataFrame(x_c).T
y_c = y_data.columns
y_c = pd.DataFrame(y_c).T
x_data.columns = list(range(len(x_data.columns)))
y_data.columns = [0]

x_data = pd.concat([x_c, x_data], axis=0)
y_data = pd.concat([y_c, y_data], axis=0)

# CatBoost参数空间替代XGBoost参数空间
CB_param_space = {
    'iterations': (100, 1000),
    'depth': (1, 10),
    'learning_rate': (0.01, 0.3),
    'l2_leaf_reg': (1, 10),
    'border_count': (5, 255)
}

# CB_param_space = {
#     'iterations': (100, 200),
#     'depth': (6, 8),
#     'learning_rate': (0.1, 0.2),
#     'l2_leaf_reg': (1, 10)
# }


RF_param_space = {
    'n_estimators': (10, 200),
    'max_depth': (1, 20),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 20)
}


def random_jitter(data, sigma=0.0001):
    jittered_data = data + np.random.normal(0, sigma, data.shape)
    return jittered_data


def add_noise(data, noise_factor=0.0001):
    noisy_data = data + noise_factor * np.random.randn(*data.shape)
    return noisy_data


def data_augmentation(data, labels, num_augmentations=5):
    augmented_datas = []
    augmented_labelss = []

    for _ in range(num_augmentations):
        augmented_data = random_jitter(np.array(data))
        augmented_data = add_noise(augmented_data)
        augmented_datas.append(augmented_data)
        augmented_labelss.append(np.ravel(labels))

    augmented_datas = np.vstack(augmented_datas)
    augmented_labelss = np.hstack(augmented_labelss)

    augmented_datas = np.vstack((data, augmented_datas))
    augmented_labelss = np.hstack((np.ravel(labels), augmented_labelss))

    return augmented_datas, augmented_labelss


def s(pre, label):
    l = []
    error_rates = np.abs((pre - label) / label) * 100
    for i, error_rate in enumerate(error_rates):
        l.append(f'{error_rate:.2f}%')
    return l


def print_data(model_name, file_name, y_test, y_pred):

    Y_test = y_test.copy()
    Y_test[model_name + "_预测"] = sorted(y_pred)
    Y_test.columns = ["真实值", model_name + "_预测"]
    Y_test["真实值"] = sorted(Y_test["真实值"].values)

    l3 = s(Y_test[model_name + '_预测'], Y_test['真实值'])

    Y_test[model_name + "_误差率"] = l3
    Y_test = Y_test.sample(len(Y_test))
    print(model_name, Y_test)
    Y_test.to_csv(model_name + "_" + file_name + "预测误差率.csv")


def mse_mae_r2(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)


    mae = mean_absolute_error(y_true, y_pred)


    r2 = r2_score(sorted(y_true[0].values), sorted(y_pred))

    return mse, mae, r2


datas = pd.DataFrame()
name_list = []
mse_list = []
mae_list = []
r2_list = []

rmse_list = []


def train_model(model, model_name, file_name, val_data, val_labels):
    model.fit(augmented_feature_datas, augmented_labelss)
    y_pred1 = model.predict(val_data)
    mse, mae, r2 = mse_mae_r2(val_labels, y_pred1)
    name_list.append(model_name + "_" + file_name)
    mse_list.append(mse)
    mae_list.append(mae * 10)
    r2_list.append(r2)

    joblib.dump(model, r"D:\博士期间所有文件汇总\陈超文章\吸附性-共热解-调研\数据试跑\数据开启\Yield+Dp\明白后集成\2\last\last-1\回归到正常参数\model\{}.pkl".format(model_name))
    print(model_name + " 均方误差 (MSE):", float(mse))
    print(model_name + " 平均绝对误差 (MAE):", float(mae * 10))
    print(model_name + " R平方 (R2):", r2)
    print_data(model_name, file_name, val_labels, y_pred1)
    if model_name == "融合模型":
        rmse = mean_squared_error(val_labels, y_pred1, squared=False)
        d = {"name": model_name, "filename": file_name, "RMSE": rmse}
        rmse_list.append(d)
    return mse, mae, r2

temp_data, test_data, temp_labels, test_labels = train_test_split(x_data, y_data, test_size=0.2, random_state=42, shuffle=True)
train_data, val_data, train_labels, val_labels = train_test_split(temp_data, temp_labels, test_size=0.25, random_state=42, shuffle=False)

print("总数据样本:", len(x_data))
print("训练集样本数:", len(train_data))
print("验证集样本数:", len(val_data))
print("测试集样本数:", len(test_data))
num_augmentations = 10
augmented_feature_datas, augmented_labelss = data_augmentation(train_data, train_labels, num_augmentations)
print("数据增强{}次 样本数量为:{}".format(num_augmentations, len(augmented_feature_datas)))



mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)


def on_step(optim_result):

    results.append(optim_result['fun'] * 0.1)
    print(f"Iteration {len(results)} - RMSE: {optim_result['fun'] * 0.1}")


def save_RMSE_csv(data, name):
    data = pd.DataFrame(data, columns=["RMSE"])
    data.to_csv(name)




# 模型训练和参数优化部分，替换XGBoost为CatBoost
results = []
RF_optimal_params = BayesSearchCV(RandomForestRegressor(), RF_param_space, n_iter=50, random_state=42, n_jobs=-1, cv=5,
                                  scoring="neg_root_mean_squared_error")
RF_optimal_params.fit(augmented_feature_datas, augmented_labelss, callback=on_step)

best_RF_params = RF_optimal_params.best_params_
print("RF最优参数:", best_RF_params)

RF_model = RandomForestRegressor(**best_RF_params)
RF_mse, RF_mae, RF_r2 = train_model(RF_model, "RF", "train", train_data, train_labels)
RF_mse, RF_mae, RF_r2 = train_model(RF_model, "RF", "test", test_data, test_labels)
RF_mse, RF_mae, RF_r2 = train_model(RF_model, "RF", "val", val_data, val_labels)
save_RMSE_csv(results, "RF_RMSE.csv")




# CatBoost模型优化
results = []
CB_optimal_params = BayesSearchCV(CatBoostRegressor(logging_level='Silent'), CB_param_space, n_iter=50, random_state=42, n_jobs=-1, cv=5,
                                  scoring="neg_root_mean_squared_error")
CB_optimal_params.fit(augmented_feature_datas, augmented_labelss, callback=on_step)
best_CB_params = CB_optimal_params.best_params_
print("CB最优参数:", best_CB_params)

CB_model = CatBoostRegressor(**best_CB_params)

CB_mse, CB_mae, CB_r2 = train_model(CB_model, "CB", "train", train_data, train_labels)
CB_mse, CB_mae, CB_r2 = train_model(CB_model, "CB", "test", test_data, test_labels)
CB_mse, CB_mae, CB_r2 = train_model(CB_model, "CB", "val", val_data, val_labels)
save_RMSE_csv(results, "CB_RMSE.csv")

RF_model = RandomForestRegressor(**best_RF_params)
CB_model = CatBoostRegressor(**best_CB_params)

# 融合模型替换XGBoost为CatBoost
ensemble_bytePair = VotingRegressor(estimators=[('RF', RF_model), ('CB', CB_model)])
# Define a range of weights
weights = np.linspace(0, 1, 50)
weight_combinations = [(w, 1 - w) for w in weights]

param_grid = {'weights': weight_combinations}

scorer = make_scorer(r2_score)


grid_search = GridSearchCV(estimator=ensemble_bytePair, param_grid=param_grid, scoring=scorer, cv=5, verbose=3)
grid_search.fit(augmented_feature_datas, augmented_labelss)


best_weights = grid_search.best_params_['weights']
print(f"网格搜索 Best Weights: {best_weights}")
pd.DataFrame(best_weights).to_csv("融合模型最优权重.csv")
ensemble_bytePair = VotingRegressor(estimators=[('RF', RF_model), ('CB', CB_model)],
                                    weights=best_weights)


# 训练和评估模型，保持原有逻辑不变
mse, mae, r2 = train_model(ensemble_bytePair, "融合模型", "train", train_data, train_labels)
mse, mae, r2 = train_model(ensemble_bytePair, "融合模型", "test", test_data, test_labels)
mse, mae, r2 = train_model(ensemble_bytePair, "融合模型", "val", val_data, val_labels)

# 将所有模型的评估结果保存至CSV文件
datas["name"] = name_list
datas["mse"] = mse_list
datas["mae"] = mae_list
datas["r2"] = r2_list
datas.to_csv("RF+CB+融合模型_mse_mae_r2.csv")
pd.DataFrame(rmse_list).to_csv("融合模型RMSE.csv")
