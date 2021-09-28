import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import plot_partial_dependence
X, y = make_friedman1()
clf = GradientBoostingRegressor(n_estimators=10).fit(X, y)
plot_partial_dependence(clf, X, [0, (0, 1)])  # doctest: +SKIP
# <...>
plt.show()  # doctest: +SKIP
