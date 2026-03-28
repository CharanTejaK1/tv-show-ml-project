from sklearn.linear_model import LogisticRegression
def train_logistic_regression(X_train, y_train):
    param_grid = {'C': [0.1,0.5, 1, 2, 5,10],'solver':['lbfgs','saga'],'penalty':['l2']}
    lr = LogisticRegression(class_weight='balanced', max_iter=3000)
    grid = GridSearchCV(
        lr, 
        param_grid, 
        cv=5, 
        scoring='f1_weighted', 
        verbose=1,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_


from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
def train_svm(X_train, y_train):
    param_grid = {'C': [3,5,7,10],'loss':['squared_hinge']}
    svm = LinearSVC(class_weight='balanced', max_iter=10000)
    grid = GridSearchCV(
        svm, 
        param_grid, 
        cv=5, 
        scoring='f1_weighted', 
        verbose=1,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_
