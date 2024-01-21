import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from sklearn.model_selection import train_test_split
import numpy as np
import  warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
        self.b = np.zeros((1, 1)) if self.bias else 0


    def train(self, X, y):
        self.weights = np.zeros((X.shape[1], 1))
        for epoch in range(self.max_epochs):
            for i in range(X.shape[0]):
                prediction = float(np.dot(X[i], self.weights) + self.b)
                self.learning_rate = float(self.learning_rate)  # Convert to a float or the appropriate numeric data type
                y = y.astype(float)  # Convert y to the same numeric data type as prediction
                x_i = np.array([X[i]])
                self.weights += self.learning_rate * (y[i] - prediction) * x_i.T
                self.b += self.learning_rate * (y[i] - prediction)
    def predict(self, X):
        if self.bias:
            X = np.c_[np.ones(X.shape[0]), X]
        print(X.shape)
        print(self.weights.shape)
        y_pred = np.dot(X, self.weights)
        #y_pred_df = pd.DataFrame(y_pred)
        print(y_pred)
    
        
        y_pred_binary = np.where(y_pred >=0, 1, -1)
        return y_pred_binary
def Perc(f1, f2, c1, c2, eta, epochs, bias):
    print("Begin..............")
    data = pd.read_excel('Dry_Bean_Dataset.xlsx', engine='openpyxl')
    
    mean_abs = abs(data.mean())
  
    data = data.fillna(mean_abs)
        
    
    # Standardize the features (z-score normalization)
    scaler = StandardScaler()
    data[["Area", "Perimeter", "MajorAxisLength","MinorAxisLength","roundnes"]] = scaler.fit_transform(data[["Area", "Perimeter", "MajorAxisLength","MinorAxisLength","roundnes"]])
    def custom_mapping(label):
        if label == c1:
            return 0
        elif label == c2:
            return 1
       
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
    print(y_test)
    import matplotlib.pyplot as plt

# Define the decision boundary plot
    def plot_decision_boundary(X, y, classifier, title, xlabel, ylabel):
    # Define plot boundaries
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a mesh grid of points
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Predict the class labels for each point in the mesh grid
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

    # Create a contour plot
        plt.contourf(xx, yy, Z, alpha=0.3)

    # Scatter plot the data points
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', marker='o')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', marker='s')

    # Set labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

    # Show the legend
        plt.legend()

    # Show the plot
        plt.show()

# Call the function to plot the decision boundary
    plot_decision_boundary(X_train, y_train, perceptron, 'Decision Boundary of Perceptron', f1, f2)
