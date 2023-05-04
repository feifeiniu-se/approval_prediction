import pymysql
import re
import string
import nltk
import numpy as np
import sklearn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bert_serving.client import BertClient
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from operator import truediv
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
from tqdm import tqdm
import datetime
import torch
from sklearn.neural_network import MLPClassifier as DNN


# 把文本中的email用<email>标签标记
def sub_email(feature_request):
    print('开始sub_email')
    result = []
    for text in tqdm(feature_request):
        emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text)
        for e in emails:
            text = text.replace(e, "<email_specification>")
        result.append(text)
    return result


# 把文本中的URL用<URL>标签标记
def sub_url(feature_request):  # 输入是feature request集合
    print('开始sub_url')
    result = []
    for text in tqdm(feature_request):
        words = text.split(" ")
        for w in words:
            if "http" in w:
                text = text.replace(w, "<url_specification>")
        result.append(text)
    return result


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# 文本预处理
def preprocess(feature_request):
    print('开始preprocess')
    result = []
    for text in tqdm(feature_request):
        text = text.lower()  # to lower case
        remove = str.maketrans('', '', string.punctuation)
        text = text.translate(remove)  # remove punctuation
        text = nltk.word_tokenize(text)  # fenci
        text = [w for w in text if not w in stopwords.words('english')]  # remove stopwords
        tagged_sent = nltk.pos_tag(text)  # 获取单词词性
        wnl = nltk.WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
        lemmas_sent = " ".join(lemmas_sent)
        result.append(lemmas_sent)
    return result  # 返回还原词性之后的句子


# 使用bert进行向量化
def bert_service(x):
    bc = BertClient(port=86500, port_out=86501, show_server_config=False)
    print('开始bert服务')
    vec = bc.encode(x)
    return vec


# 判断x时间是否比y早，若早返回true，否则返回false
def compareTime(x, y):
    x_y, x_m, x_d = x.split('-')
    y_y, y_m, y_d = y.split('-')
    a = datetime.date(int(x_y), int(x_m), int(x_d))
    b = datetime.date(int(y_y), int(y_m), int(y_d))
    return a.__le__(b)


def extract_words(TEXT):
    result = []
    words = word_tokenize(TEXT)
    tags = nltk.pos_tag(words)
    for i, element in enumerate(tags):
        if (element[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
            result.append(element[0])
    return result

def nltk_sentence_bleu(hypothesis, reference, order=4):
    cc = SmoothingFunction()
    if len(reference) < 4:
        return 0
    try:
        score = nltk.translate.bleu([reference], hypothesis, smoothing_function=cc.method1)
    except Exception as e:
        print('reference: ', reference)
        print('hypothesis: ', hypothesis)
        print(e)
    # print(score)
    return score

#request是从数据库检索出来的按照时间顺序排好序的，第i个是待求的
def sim_request(request, index):
    summary_query = request[index][1]
    content_query = request[index][3]
    sim_table = []
    it = 0
    while(it<index):

        it += 1

    pass


def process_content(mycursor):
    items = ['zk1', 'oscarmcmaster', 'keepass', 'winmerge', 'skim-app', 'adempiere', 'texstudio', 'gallery',
             'phpmyadmin', 'mumble', 'arianne', 'phpgedview', 'popfile', 'scintilla', 'omegat']
    # items = ['zk1']

    # 从数据库中提取feature_request文本信息
    x_train, x_test, y_train, y_test, senti_train, senti_test = [], [], [], [], [], []
    can_vec, cannot_vec = [], []
    # 所有项目分别按照时间顺序排序，然后截取前边的0.8作为训练集，后边的0.2作为测试集
    for item in items:
        print(f'现在开始处理的项目为{item}')

        # 所有的文本
        mycursor.execute(
            "SELECT project_name, summary, created_time, content, approval_class,request_id,id, sentiment_predict FROM feature_request where project_name='" + item + "' and created_time IS NOT NULL and (approval_class='Can be implemented' or approval_class='Can not be implemented') ORDER BY created_time")
        myresult = mycursor.fetchall()

        request = list(myresult[:])  # 用来选择训练的数据个数
        summary, content = [], []
        sentiment, y = [], []
        can, cannot = [], []

        dict_ = {"Can be implemented": 0, "Can not be implemented": 1}
        sentiment_classes = set(['Neutral','Negative','Positive'])
        dict_senti = {c: np.identity(len(sentiment_classes))[i, :] for i, c in enumerate(sentiment_classes)}
        print('开始存储数据')

        for i, tmp_tuple in enumerate(tqdm(request)):
            summary.append(tmp_tuple[1].replace('\n', ' '))
            content.append(tmp_tuple[3].replace('\n', ' '))
            # request_content.append((tmp_tuple[1] + ' ' + tmp_tuple[3] + ' ').replace('\n', ' '))
            y.append(dict_[tmp_tuple[4]])
            sentiment.append(dict_senti[tmp_tuple[7]])
            can, cannot = sim_request(request, i)


        # 将email和url替换掉
        content = sub_email(content)
        content = sub_url(content)
        content = preprocess(content)
        feature_request = []
        for i, tmp_tuple in enumerate(tqdm(content)):
            feature_request.append(summary[i]+' '+content[i])

        for i in range(len(request)):
            if i < int(len(request) * 0.8):
                x_train.append(feature_request[i])
                y_train.append(y[i])
                senti_train.append(sentiment[i])
            else:
                x_test.append(feature_request[i])
                y_test.append(y[i])
                senti_test.append(sentiment[i])


    #tfidf ngram 2-10 向量化
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(2, 5), encoding='utf-8')
    vectorizer.fit(x_train)
    vec_train = vectorizer.transform(x_train).toarray()
    vec_test = vectorizer.transform(x_test).toarray()
    print(vec_train.shape, vec_test.shape)

    # # bert向量化
    # print('使用bert向量化')
    # vec_train=bert_service(x_train)
    # vec_test=bert_service(x_test)
    # print('x_train个数为{}，y_train个数为{},x_test个数为{}，y_test个数为{}'.format(len(vec_train),len(y_train),len(vec_test),len(y_test)))

    # vec_train = MinMaxScaler().fit_transform(vec_train)
    # vec_test = MinMaxScaler().fit_transform(vec_test)

    # # # 拼接情感
    # vec_train = np.concatenate((vec_train, senti_train), axis=1)
    # vec_test = np.concatenate((vec_test, senti_test), axis=1)
    # print(vec_train.shape, vec_test.shape)

    return vec_train, y_train, vec_test, y_test


def model_bayes(x_train, y_train):
    # 参考连接：https://blog.csdn.net/dingustb/article/details/81319948
    model = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    model.fit(x_train, y_train)
    return model

def model_dense(vec_train,y_train):
    ''' sklearn神经网络参数介绍下方链接:
    1.https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html?highlight=mlpclassifier#sklearn.neural_network.MLPClassifier
    2.参考博客:https://blog.csdn.net/gracejpw/article/details/103198598?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control
    '''
    # 默认三层全连接,仅有一个隐藏层
    model = DNN(hidden_layer_sizes=(128,),max_iter=500,random_state=420, early_stopping=True)
    model.fit(vec_train,y_train)
    return model

def P_R_F(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_true, y_pred)
    # print("confusion_matrix: ", confusion_matrix)

    # 计算第i类的precision, recall and F
    diag = np.diag(confusion_matrix)  # 取对角线上的值
    raw_sum = np.sum(confusion_matrix, axis=1)  # 计算每一行的和
    each_acc = np.nan_to_num(truediv(diag, raw_sum))
    # print("recall: ", each_acc)

    column_sum = np.sum(confusion_matrix, axis=0)  # 计算每一列的和
    each_precision = np.nan_to_num(truediv(diag, column_sum))
    # print("precision: ", each_precision)

    F = []
    i = 0
    while i < len(each_precision):
        F.append(2 * each_precision[i] * each_acc[i] / (each_precision[i] + each_acc[i]))
        i = i + 1
    return each_acc, each_precision, F


if __name__ == '__main__':
    db = pymysql.connect(
        host="172.29.7.248",
        port=3306,
        user="root",
        password="1234",
        database="sourceforge"
    )
    mycursor = db.cursor()
    file_path = './compare.txt'  # 存储每个模型最终打印的结果
    file = open(file_path, 'w')
    # 得到处理之后的文本并转化tfidf向量
    x_train, y_train, x_test, y_test = process_content(mycursor)

    model = model_bayes(x_train, y_train)
    y_predict = model.predict(x_test)
    print('训练集分数：', model.score(x_train, y_train), '测试集分数', model.score(x_test, y_test))
    file.writelines('训练集分数：{},测试集分数:{}\n'.format(model.score(x_train, y_train), model.score(x_test, y_test)))
    y_predict = list(y_predict)

    score = 0
    for i in range(len(y_predict)):
        if y_predict[i] == y_test[i]:
            score = score + 1
    score = score / len(y_test)
    each_recall, each_precision, each_f = P_R_F(y_test, y_predict)

    print(score)
    print(each_precision)
    print(each_recall)
    print(each_f)

    total = 0
    for i in y_train:
        total += i
    print(total)
    print(len(y_train))
    total = 0
    for i in y_test:
        total += i
    print(total)
    print(len(y_test))
