import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import det_curve, DetCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X, y = make_classification(n_samples=1000, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)
clf = SVC(random_state=0).fit(X_train, y_train)
y_pred = clf.decision_function(X_test)
fpr, fnr, _ = det_curve(y_test, y_pred)
display = DetCurveDisplay(
    fpr=fpr, fnr=fnr, estimator_name="SVC"
)
display.plot()
# <...>
plt.show()
