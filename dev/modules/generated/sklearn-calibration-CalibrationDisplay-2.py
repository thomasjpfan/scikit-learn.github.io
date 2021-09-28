import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibrationDisplay
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)
# LogisticRegression(random_state=0)
y_prob = clf.predict_proba(X_test)[:, 1]
disp = CalibrationDisplay.from_predictions(y_test, y_prob)
plt.show()
