import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib


def evaluate_model(test_data_path, model_path):
    # 读取测试数据
    X_test = pd.read_csv(f'{test_data_path}/X_test.csv')
    y_test = pd.read_csv(f'{test_data_path}/y_test.csv')

    # 加载模型
    model = joblib.load(model_path)

    # 预测并评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Model evaluation completed. Mean Squared Error: {mse}")


if __name__ == "__main__":
    evaluate_model('split', 'model/linear_regression_model.pkl')
