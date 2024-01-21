import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas.core.common import SettingWithCopyWarning
import  warnings
# Suppress the SettingWithCopyWarning
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
class Adaline:
    def __init__(self, learning_rate, max_epochs,bias):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.bias = bias

    def train(self, X, y):
        if self.bias:
            X = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.zeros(X.shape[1])
        for epoch in range(self.max_epochs):
            for i in range(X.shape[0]):
                prediction = np.dot(X[i], self.weights)
                if prediction >= 0:
                    prediction = 1
                else:
                    prediction = -1
                if prediction != y[i]:
                    self.learning_rate = float(self.learning_rate)
                    y = y.astype(float)
                    driv = -(2 / len(X)) * np.dot(X[i], (y[i] - prediction))
                    self.weights += self.learning_rate * driv

    def predict(self, X):
        if self.bias:
            X = np.c_[np.ones(X.shape[0]), X]
        y_pred = np.dot(X, self.weights)
        y_pred_binary = np.where(y_pred >= 0, 1, -1)
        return y_pred_binary
def Perc2(f1, f2, c1, c2, eta, epochs, bias):
    print("Begin..............")
    data = pd.read_excel('Dry_Bean_Dataset.xlsx', engine='openpyxl')

    def custom_mapping(label):
        if label == c1:
            return 0
        elif label == c2:
            return 1
        

    condition = (data["Class"] == c1) | (data["Class"] == c2)
    selected_rows = data[condition]
    selected_rows.loc[:, 'Class'] = selected_rows['Class'].apply(custom_mapping)

    if f1 not in selected_rows.columns or f2 not in selected_rows.columns:
        print("Invalid column names.")
        return

    selected_rows = selected_rows[[f1, f2, "Class"]]
    x = selected_rows.drop(columns=["Class"]).values
    y = selected_rows["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
   
    # perceptron = Perceptron(eta, epochs,bias)
    adaline = Adaline(eta, epochs, bias)
    adaline.train(X_train, y_train)
    y_pred = adaline.predict(X_test)
    total_predictions = len(y_test)
    correct_predictions = np.sum(y_pred.flatten() == y_test)
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Adaline Accuracy: {accuracy:.2f}%")
    import matplotlib.pyplot as plt

# Assuming 'f1' and 'f2' are the features you want to plot
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='BOMBAY', marker='o')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='CALI', marker='s')
    plt.scatter(X_train[y_train == 2][:, 0], X_train[y_train == 2][:, 1], label='SIRA', marker='x')

    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.legend()
    plt.title('Scatter Plot of Data')
    plt.show()