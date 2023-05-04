from sklearn import svm
# from thundersvm import SVC
# from cuml.svm import SVC

# def SVM_model(x_train, y_train):

#     model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#                     decision_function_shape='ovo', degree=3, gamma='auto', kernel='sigmoid',
#                     max_iter=-1, probability=False, random_state=None, shrinking=True,
#                     tol=0.001, verbose=False)
#     model.fit(x_train, y_train)
#     return model#nff原来的
# def SVM_model(x_train, y_train):

#     model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#                     decision_function_shape='ovo', degree=3, gamma=0.1, kernel='rbf',
#                     max_iter=10, probability=False, random_state=None, shrinking=True,
#                     tol=0.001, verbose=False)
#     model.fit(x_train, y_train)
#     return model #学妹新写的

def SVM_model(x_train, y_train):
    # {1 : 3}
    model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
                    max_iter=30, probability=False, random_state=None, shrinking=True,
                    tol=0.0001, verbose=False)
    model.fit(x_train, y_train)
    return model #ch调参实验

# def SVM_GPU(x_train, y_train):
#     model = SVC(kernel='linear')
#     model.fit(x_train, y_train)
#     return model