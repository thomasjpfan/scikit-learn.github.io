import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
clf = LogisticRegression()
clf.fit(X_train, y_train)
# LogisticRegression()
y_pred = clf.predict_proba(X_test)[:, 1]
PrecisionRecallDisplay.from_predictions(
   y_test, y_pred)
# <...>
plt.show()
