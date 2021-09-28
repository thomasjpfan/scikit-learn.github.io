import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_det_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X, y = make_classification(n_samples=1000, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)
clf = SVC(random_state=0).fit(X_train, y_train)
plot_det_curve(clf, X_test, y_test)  # doctest: +SKIP
# <...>
plt.show()
