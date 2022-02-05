from sklearn.tree import DecisionTreeClassifier
import numpy as np


class MyAdaBoost:

    def __init__(self):
        self.T = None
        self.alphas = None
        self.error = None
        self.stumps = []

    def fit(self, T, X, Y):
        self.T = T
        self.alphas = []
        weight = np.ones(len(Y)) / len(Y)

        # La costruzione del weak learner è basata sull'utilizzo di sklearn attraverso decisiontreeClassifier e i
        # suoi metodi
        for t in range(0, T):
            stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            stump.fit(X, Y, sample_weight=weight)
            y_pred = stump.predict(X)

            self.stumps.append(stump)
            self.error = (sum(weight[(y_pred != Y)]))
            alpha = 1 / 2 * np.log((1 - self.error) / self.error)
            self.alphas.append(alpha)

            new_weight = weight * np.exp(-alpha * Y * y_pred)
            new_weight /= sum(new_weight)  # Normalizing
            weight = new_weight

    def predict(self, X):
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        finClass = np.sign(np.dot(self.alphas, stump_preds))

        return finClass
