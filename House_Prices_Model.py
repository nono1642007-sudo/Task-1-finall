import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("house_prices_cleaned.csv")
price_column = next(
    column for column in df.columns
    if column.strip().lower()=="price"
)

y = df[price_column]

x = df.drop(columns=[price_column])
x = x.select_dtypes(include=["number"])

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE:" , rmse)