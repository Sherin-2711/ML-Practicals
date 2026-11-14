import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Read the dataset
df = pd.read_csv("emails.csv")
df = df.drop_duplicates()

# 2.Separate features and target
target = "Prediction"
x = df.drop(columns = [target,"Email No."], errors = "ignore")
y = df[target]

# 3. Split dataset
x_train,x_test,y_train, y_test = train_test_split(
    x,y,test_size = 0.2, stratify=y
)

# 4. Normalize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 5. Train and compare classifiers
models = {
    "KNN": KNeighborsClassifier(n_neighbors = 5),
    "SVM": SVC(kernel = "linear")
}


for name, model in models.items():
    model.fit(x_train,y_train)
    pred = model.predict(x_test)

    print("\n",name)
    print("Accuracy: ",accuracy_score(y_test,pred))
    print("Confusion Matrix: ",confusion_matrix(y_test,pred))
    print("Classification report: ", classification_report(y_test,pred,zero_division=0))