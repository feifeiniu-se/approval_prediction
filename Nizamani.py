from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier as DNN

def MNB_model(x_train, y_train):
    # 参考连接：https://blog.csdn.net/dingustb/article/details/81319948
    model = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    # model = GaussianNB()
    model.fit(x_train, y_train)
    return model

def NN_model(x_train,y_train):
    ''' sklearn神经网络参数介绍下方链接:
    1.https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html?highlight=mlpclassifier#sklearn.neural_network.MLPClassifier
    2.参考博客:https://blog.csdn.net/gracejpw/article/details/103198598?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control
    '''
    # 默认三层全连接,仅有一个隐藏层
    model = DNN(hidden_layer_sizes=(128,),max_iter=500,random_state=420, early_stopping=True)
    model.fit(x_train,y_train)
    return model