# ============================================================
# Gradient Boosting From Scratch + Scikit-learn Comparison
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# Utility Functions
# ============================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ============================================================
# Gradient Boosting From Scratch (Binary Classification)
# ============================================================

class GradientBoostingFromScratch:
    """
    Gradient Boosting for Binary Classification using
    decision tree regressors and log-loss optimization.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.init_prediction = None

    def fit(self, X, y):
        # Initialize predictions with log-odds
        positive_ratio = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.init_prediction = np.log(positive_ratio / (1 - positive_ratio))

        y_pred = np.full(len(y), self.init_prediction)

        for _ in range(self.n_estimators):
            # Compute pseudo-residuals (gradient of log-loss)
            residuals = y - sigmoid(y_pred)

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            update = tree.predict(X)
            y_pred += self.learning_rate * update

            self.models.append(tree)

    def predict_proba(self, X):
        y_pred = np.full(X.shape[0], self.init_prediction)

        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X)

        probs = sigmoid(y_pred)
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def feature_importances(self):
        importances = np.zeros(self.models[0].n_features_in_)
        for tree in self.models:
            importances += tree.feature_importances_
        return importances / len(self.models)


# ============================================================
# Data Generation
# ============================================================

X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# Baseline Models
# ============================================================

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)
log_reg_acc = accuracy_score(y_test, log_reg_preds)

tree_clf = DecisionTreeClassifier(max_depth=3)
tree_clf.fit(X_train, y_train)
tree_preds = tree_clf.predict(X_test)
tree_acc = accuracy_score(y_test, tree_preds)


# ============================================================
# Gradient Boosting From Scratch
# ============================================================

gb_scratch = GradientBoostingFromScratch(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

gb_scratch.fit(X_train, y_train)
scratch_preds = gb_scratch.predict(X_test)
scratch_acc = accuracy_score(y_test, scratch_preds)


# ============================================================
# Scikit-learn Gradient Boosting
# ============================================================

gb_sklearn = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

gb_sklearn.fit(X_train, y_train)
sklearn_preds = gb_sklearn.predict(X_test)
sklearn_acc = accuracy_score(y_test, sklearn_preds)


# ============================================================
# Results
# ============================================================

print("\nModel Performance Comparison")
print("=" * 40)
print(f"Logistic Regression Accuracy      : {log_reg_acc:.4f}")
print(f"Decision Tree Accuracy            : {tree_acc:.4f}")
print(f"Gradient Boosting (From Scratch)  : {scratch_acc:.4f}")
print(f"Gradient Boosting (Scikit-learn)  : {sklearn_acc:.4f}")


# ============================================================
# Feature Importance Comparison
# ============================================================

scratch_importance = gb_scratch.feature_importances()
sklearn_importance = gb_sklearn.feature_importances_

plt.figure(figsize=(10, 5))
plt.bar(range(len(scratch_importance)), scratch_importance, alpha=0.6, label="From Scratch")
plt.bar(range(len(sklearn_importance)), sklearn_importance, alpha=0.6, label="Scikit-learn")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Feature Importance Comparison")
plt.legend()
plt.tight_layout()
plt.show()
