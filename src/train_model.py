# src/train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def train_model(train_data_path, model_save_path):
    # 读取训练数据
    X_train = pd.read_csv(f'{train_data_path}/X_train.csv')
    y_train = pd.read_csv(f'{train_data_path}/y_train.csv')

    # 创建并训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 保存模型
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    train_model('split', 'model/linear_regression_model.pkl')


