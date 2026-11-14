import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("uber.csv")
df = df.dropna()
df = df.drop_duplicates()

df = df.drop(columns=["Unamed: 0", "key"], errors ="ignore")

df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"], errors = "coerce", utc = True
)

df = df.dropna(subset=["pickkup_datetime"])

df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month
df = df.drop(columns = ["pickup_datetime"])

df = df[
    (df["fare_amount"] > 0)&
    df["pickup_longitude"].between(-180,180) &
    df["dropoff_longitude"].between(-180,180) &
    df["pickup_latitude"].between(-90,90) &
    df["dropoff_latitude"].between(-90,90)
]

Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)

IQR = Q3 - Q1

outliers = df[
    (df["fare_amount"] < Q1 - 1.5 * IQR) |
    (df["fare_amount"] > Q3 + 1.5 * IQR)
]

print("Fare outliers: ", outliers)

df = df[
    df["fare_amount"].between(
        max(0,Q1-1.5*IQR), Q3+1.5*IQR
    )
]

print(df.corr(numeric_only=True)["fare_amount"].sort_values())

sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation heatmap")
plt.show()

x = df.drop(colums=["fare_amount","passenger_count"], errors="ignore")
y = df["fare_amount"]

x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.2,random_state = 42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random forest": RandomForestRegressor(
        n_estimators=100, random_state= 42, n_jobs=1
    )
}

for name, model in models.items():
    model.fit(x_train,y_train)
    pred = model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test,pred))
    print(name)
    print("R2 Score: ", round(r2_score(y_test,pred),4))
    print("RMSE: ", round(rmse,4))
    print("MAB: ", round(mean_absolute_error(y_test,pred),4))

