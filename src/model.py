import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_iris_data():
    """Loads the Iris dataset and returns features and target."""
    from sklearn.datasets import load_iris
    iris = load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names

def split_data(X, y):
    """Splits data into train and test sets."""
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_logistic_regression(X_train, y_train):
    """Trains a Logistic Regression model."""
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, max_depth=3):
    """Trains a Decision Tree model."""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, target_names):
    """Evaluates a model and prints metrics."""
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=target_names))

# Example usage if running as a script:
if __name__ == "__main__":
    X, y, feature_names, target_names = load_iris_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Logistic Regression
    log_reg = train_logistic_regression(X_train, y_train)
    print("Logistic Regression Results:")
    evaluate_model(log_reg, X_test, y_test, target_names)
    
    # Decision Tree
    dtree = train_decision_tree(X_train, y_train)
    print("\nDecision Tree Results:")
    evaluate_model(dtree, X_test, y_test, target_names)
