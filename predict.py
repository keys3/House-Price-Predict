from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import os
csv_file_path = os.path.join('Data/Raw/1553768847-housing.csv')
df = pd.read_csv(csv_file_path)
# Handling missing values in 'total_bedrooms' using median imputation
imputer = SimpleImputer(strategy='median')

# Encoding the categorical 'ocean_proximity' feature
encoder = OneHotEncoder()

# Defining feature columns and target variable
features = df.drop("median_house_value", axis=1)
target = df["median_house_value"]

# Numerical columns
num_cols = features.select_dtypes(include=['float64', 'int64']).columns
# Categorical columns
cat_cols = ['ocean_proximity']

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', imputer, num_cols),
        ('cat', encoder, cat_cols)])

# Creating the pipeline with a Linear Regression model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(mse)
