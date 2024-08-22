from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

RAW_DATA_PATH = 'Data/Raw/1553768847-housing.csv'
PROCESSED_DATA_PATH = 'Data/Processed/house_prices_processed.csv'


# 1. 读取原始数据
def load_data(file_path):
    """读取CSV文件并返回DataFrame."""
    data = pd.read_csv(file_path)
    return data


# 数据清洗
def clean_data(data):
    """清洗数据：处理缺失值、异常值等."""
    # 例如：删除缺失值较多的行
    # 区分数值列和类别列
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # 对数值数据使用中位数填充
    numeric_imputer = SimpleImputer(strategy='median')
    data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])

    # 对类别数据使用最常见值填充
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])
    return data


# 3. 特征工程
def feature_engineering(data):
    """特征工程：创建新的特征或转换现有特征."""
    # 例如：可以根据房龄创建一个新特征，表示房屋是否为新房（小于30年）
    data['IsNew'] = data['housing_median_age'] < 30
    return data


# 数据标准化
def standardize_features(data):
    """标准化特征值."""
    df = pd.DataFrame(data)

    # 对数值特征进行标准化（假设所有特征都是数值型的）
    # 数值特征列
    numerical_features = [
        'longitude', 'latitude', 'housing_median_age',
        'total_rooms', 'total_bedrooms', 'population',
        'households', 'median_income', 'median_house_value']
    # 类别特征列
    categorical_features = ['ocean_proximity']
    # 对数值特征进行标准化
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(df[numerical_features])
    # 对类别特征进行独热编码
    one_hot_encoder = OneHotEncoder(sparse=False)
    encoded_categorical_data = one_hot_encoder.fit_transform(df[categorical_features])
    # 将独热编码后的数据转换为 DataFrame，并与原始 DataFrame 合并
    encoded_categorical_df = pd.DataFrame(
        encoded_categorical_data,
        columns=one_hot_encoder.get_feature_names_out(categorical_features)
    )

    # 合并编码后的类别特征与原始数据
    df = pd.concat([df.drop(columns=categorical_features), encoded_categorical_df], axis=1)
    return df


# 5. 保存处理后的数据
def save_data(df, file_path):
    """保存处理后的数据到CSV文件."""
    df.to_csv(file_path, index=False)
    print(f"Processed data saved to {file_path}")


# 主预处理函数
def preprocess_data():
    # 加载数据
    data = load_data(RAW_DATA_PATH)

    # 数据清洗
    data = clean_data(data)

    # 特征工程
    data = feature_engineering(data)

    # 标准化
    df = standardize_features(data)

    # 保存处理后的数据
    save_data(df, PROCESSED_DATA_PATH)


if __name__ == '__main__':
    preprocess_data()
