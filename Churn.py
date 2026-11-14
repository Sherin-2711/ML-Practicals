import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# 1. Read dataset
df = pd.read_csv("Churn_Modelling.csv")
# 2. Remove irrelevant columns
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")
# Separate features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]
# Convert categorical columns into numerical columns
X = pd.get_dummies(X, columns=["Geography", "Gender"], drop_first=True)
# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42, stratify=y
)
# Normalize data (fit only on training data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 4. Initial neural network
model1 = MLPClassifier(
 hidden_layer_sizes=(16,),
 max_iter=300,
 random_state=42
)
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)
print("Initial Model Accuracy:", accuracy_score(y_test, pred1))
# Improved neural network
model2 = MLPClassifier(
 hidden_layer_sizes=(32, 16),
 activation="relu",
 solver="adam",
 alpha=0.01,
 learning_rate_init=0.001,
 max_iter=500,
 early_stopping=True,
 random_state=42
)
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)
# 5. Evaluate improved model
print("\nImproved Model Accuracy:", accuracy_score(y_test, pred2))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred2))
print("Classification Report:\n",
 classification_report(y_test, pred2, zero_division=0))