from sklearn.preprocessing import StandardScaler
import numpy as np
def test():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # 创建标准化对象
    scaler = StandardScaler()

    # 对数据进行标准化
    standardized_data = scaler.fit_transform(data)

    print("Standardized Data:")
    print(standardized_data)
if __name__ == '__main__':
    test()
