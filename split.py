import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 定义路径
PROCESSED_DATA_PATH = 'Data/Processed/house_prices_processed.csv'
SPLIT_DIR = 'src/split'

# 1. 加载处理后的数据
data = pd.read_csv(PROCESSED_DATA_PATH)

# 2. 分离特征和目标变量
X = data.drop(columns=['median_house_value'])  # 假设'Price'是目标变量,这将删除'Price'列，并保留其他所有列作为特征
y = data['median_house_value']

# 3. 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. 保存拆分后的数据
X_train.to_csv(os.path.join(SPLIT_DIR, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(SPLIT_DIR, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(SPLIT_DIR, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(SPLIT_DIR, 'y_test.csv'), index=False)

print("Data successfully split and saved to 'split/' directory.")
