import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
clf = SVC(random_state=0).fit(X_train, y_train)
RocCurveDisplay.from_estimator(
   clf, X_test, y_test)
# <...>
plt.show()
