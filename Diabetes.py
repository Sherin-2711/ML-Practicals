import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, recall_score,precision_score)

df = pd.read_csv("diabetes.csv")
df = df.drop_duplicates()

x = df.drop("Outcome", axis = 1)
y = df["Outcome"]

cols = ["Diabetes", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols:
    if col in x.columns:
        x[col] = x[col].replace(0,float("nan"))
        x[col] = x[col].fillna(x[col].median())

x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.2,stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = KNeighborsClassifier(n_neighbors = 5)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print("Accurcy: ", accuracy)
print("Error rate: ", 1 - accuracy)
print("Confusion Matrix: ",confusion_matrix(y_test,y_pred))
print("Precision: ", precision_score(y_test, y_pred, zero_division = 0))
print("Recall: ", recall_score(y_test, y_pred, zero_division = 0))