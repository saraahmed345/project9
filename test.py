import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from sklearn.model_selection import train_test_split
import numpy as np
import  warnings

# Suppress the SettingWithCopyWarning
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)


    # Your code that generates the warning

# Warnings are back to their previous state outside the context manager

import numpy as np

class Perceptron:
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
                prediction = float(np.dot(X[i], self.weights))
                if prediction >= 0:
                    prediction = 1
                else:
                    prediction = -1
                if prediction != y[i]:
                    self.learning_rate = float(self.learning_rate)  # Convert to a float or the appropriate numeric data type
                    y = y.astype(float)  # Convert y to the same numeric data type as prediction
                    self.weights += self.learning_rate * (y[i] - prediction) * X[i]
    def predict(self, X):
        if self.bias:
            X = np.c_[np.ones(X.shape[0]), X]
        y_pred = np.dot(X, self.weights)
        print(y_pred)
        y_pred_binary = np.where(y_pred >= 0, 1, -1)
        return y_pred_binary
def Perc(f1, f2, c1, c2, eta, epochs, bias):
    print("Begin..............")
    data = pd.read_excel('Dry_Bean_Dataset.xlsx', engine='openpyxl')

    def custom_mapping(label):
        if label == 'BOMBAY':
            return 0
        elif label == 'CALI':
            return 1
        elif label == 'SIRA':
            return 2
        # Add more conditions for other labels as needed

    condition = (data["Class"] == c1) | (data["Class"] == c2)
    selected_rows = data[condition]
    selected_rows.loc[:, 'Class'] = selected_rows['Class'].apply(custom_mapping)

    # Verify that f1 and f2 are valid column names in your dataset
    if f1 not in selected_rows.columns or f2 not in selected_rows.columns:
        print("Invalid column names.")
        return

    selected_rows = selected_rows[[f1, f2, "Class"]]
    x = selected_rows.drop(columns=["Class"]).values
    y = selected_rows["Class"].values
    print(x)
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    perceptron = Perceptron(eta, epochs,bias)
    print("done 1")
    perceptron.train(X_train, y_train)
    print("done 2")
    correct_predictions = 0
    total_predictions = len(X_test)
    for i in range(total_predictions):
        x = X_test[i]
        prediction = perceptron.predict(x)
        if prediction == y_test[i]:
            correct_predictions += 1
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy: {accuracy:.2f}%")
