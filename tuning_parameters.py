from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score

# MAIN: Data reading used to tune pca

train_set = pd.read_csv("train_set.csv")
test_set = pd.read_csv("test_set.csv")
df_development = pd.read_csv("C:/Users/emanu/OneDrive/Desktop/dsl_data/development.csv")
train_label = []
i = 1

function = pd.DataFrame(columns=['x', 'y_svm', 'y_random_forest'])

# 2) READING OF LABELS

for ind, row in df_development.iterrows():
    lab = row['action'] + row['object']
    train_label.append(lab)

# 3) CYCLE ON EVERY NUMBER OF n_components BETWEEN 1 AND 70 WITH VALIDATION

while i < 70:
    pca = PCA(n_components=i)  # inserire valore valido pca
    train_set = np.array(train_set)
    test_set = np.array(test_set)
    pca_scaled = pca.fit_transform(train_set)
    pca_scaled_test = pca.transform(test_set)
    X_train, X_test, y_train, y_test = train_test_split(pca_scaled, np.array(train_label), test_size=0.15, stratify=train_label)

    clf2 = RandomForestClassifier(n_estimators=1500)
    clf2.fit(X_train, y_train)
    ypred2 = clf2.predict(X_test)
    result2 = pd.DataFrame(ypred2)
    acc2 = accuracy_score(y_test, np.array(ypred2)) * 100
    p2, r2, f12, s2 = precision_recall_fscore_support(y_test, ypred2)
    print(f"RFC pca = {i} accuracy: {acc2}")
    print(f"precision = {p2.mean()} %")
    print(f"recall = {r2.mean()} %")

    clf1 = svm.SVC(C=10, kernel='rbf', gamma=0.1)
    clf1.fit(X_train, y_train)
    ypred1 = clf1.predict(X_test)
    ypred1 = pd.DataFrame(ypred1)
    acc1 = accuracy_score(y_test, np.array(ypred1)) * 100
    p1, r1, f1, s1 = precision_recall_fscore_support(y_test, ypred1)
    print(f"SVM pca = {i} accuracy: {acc1}")
    print(f"precision = {p1.mean()} %")
    print(f"recall = {r1.mean()} %")

    # we used param_grid to find the best hyperparameters for svc

    '''
    param_grid = {'C': [10,15,20], 'gamma': [0.2, 0.1, 0.4]}

    svm = SVC(kernel='rbf')

    grid_search = GridSearchCV(svm, param_grid, cv=5)

    grid_search.fit(X_train, y_train)

    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    '''

    function.loc[len(function)] = [i, acc1, acc2]

    if i <= 10:
        i = i + 1
    else:
        i = i + 2

    acc_svm = cross_val_score(clf1, X_train, y_train, cv=5, scoring='accuracy')
    print(acc_svm)

# VALUES HAS BEEN SAVED TO PLOT THE RESULTS
function.to_csv("pca_accuracy_models.csv")
