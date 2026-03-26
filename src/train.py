from sklearn.linear_model import LogisticRegression
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        C=2,
        max_iter=3000,
        class_weight='balanced',
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    return model


from sklearn.svm import LinearSVC
def train_svm(X_train, y_train):
    model = LinearSVC(
        C=1,
        class_weight='balanced',
        max_iter=10000
    )
    model.fit(X_train, y_train)
    return model