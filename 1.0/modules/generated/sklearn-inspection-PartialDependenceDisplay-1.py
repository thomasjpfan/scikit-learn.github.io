import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
X, y = make_friedman1()
clf = GradientBoostingRegressor(n_estimators=10).fit(X, y)
PartialDependenceDisplay.from_estimator(clf, X, [0, (0, 1)])
# <...>
plt.show()
