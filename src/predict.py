# src/predict.py
import pandas as pd
import joblib


def predict(input_data_path, model_path, output_path):
    # 读取输入数据
    input_data = pd.read_csv(input_data_path)

    # 加载模型
    model = joblib.load(model_path)

    # 生成预测
    predictions = model.predict(input_data)

    # 保存预测结果
    pd.DataFrame(predictions, columns=['PredictedPrice']).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    predict('data/new_data.csv', 'model/linear_regression_model.pkl', 'predictions/predicted_prices.csv')