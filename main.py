# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Load data
data = pd.read_csv("data.csv") # Replace with your data file name
X = data.iloc[:, :-1].values # Features
y = data.iloc[:, -1].values # Labels

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision tree
dt = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
dt.fit(X_train, y_train) # Train the model
y_pred_dt = dt.predict(X_test) # Predict on test set
cm_dt = confusion_matrix(y_test, y_pred_dt) # Confusion matrix
acc_dt = accuracy_score(y_test, y_pred_dt) # Accuracy score

# Logistic regression
lr = LogisticRegression(solver="liblinear", random_state=42)
lr.fit(X_train, y_train) # Train the model
y_pred_lr = lr.predict(X_test) # Predict on test set
cm_lr = confusion_matrix(y_test, y_pred_lr) # Confusion matrix
acc_lr = accuracy_score(y_test, y_pred_lr) # Accuracy score

# Print results
print("Decision tree confusion matrix:")
print(cm_dt)
print("Decision tree accuracy: {:.2f}%".format(acc_dt*100))
print("Logistic regression confusion matrix:")
print(cm_lr)
print("Logistic regression accuracy: {:.2f}%".format(acc_lr*100))

# Plot decision tree
from sklearn.tree import plot_tree
plt.figure(figsize=(10, 8))
plot_tree(dt, feature_names=data.columns[:-1], class_names=data.columns[-1], filled=True)
plt.show()
