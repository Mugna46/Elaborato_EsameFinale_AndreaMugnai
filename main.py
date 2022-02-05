import numpy as np
import pandas as pd
from Adaboost_Impl import MyAdaBoost
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.preprocessing import OneHotEncoder


def main():
    data = input("Quale Dataset vuoi scegliere?")
    if data == "1":
        ds = pd.read_csv("bank-additional-full.csv", sep=";")
    else:
        ds = pd.read_csv("bank-full.csv", sep=";")

    le = preprocessing.LabelEncoder()
    for i in range(0, ds.shape[1] - 1):
        ds.iloc[:, i] = le.fit_transform(ds.iloc[:, i])

    X = ds.iloc[:, 0:ds.shape[1] - 1]  # Data
    y = ds.iloc[:, ds.shape[1] - 1]  # Target
    y = np.where(y == "no", -1, 1)

    """
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(X)
    X = enc.transform(X).toarray()
    """

    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.33)

    ab = MyAdaBoost()
    ab.fit(30, X_Train, y_Train)
    y_pred = ab.predict(X_Test)
    y_Test = y_Test.astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y_Test, y_pred)
    print(metrics.auc(fpr, tpr))
    print("ACCURACY:", metrics.accuracy_score(y_Test, y_pred))

    clf = AdaBoostClassifier(n_estimators=30, algorithm="SAMME")
    clf.fit(X_Train, y_Train)
    y_pred2 = clf.predict(X_Test)
    fpr1, tpr1, thresholds = metrics.roc_curve(y_Test, y_pred2)
    print(metrics.auc(fpr1, tpr1))
    print("ACCURACY:", metrics.accuracy_score(y_Test, y_pred2))


if __name__ == "__main__":
    main()
