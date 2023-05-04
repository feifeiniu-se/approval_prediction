import nltk
import string
import re
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from spellchecker import SpellChecker
from tqdm import tqdm
# from spellchecker import SpellChecker
from operator import truediv
import numpy as np
import pymysql
from pymysql.converters import escape_string

# https://www.cnblogs.com/shiningrain3/p/13856645.html预处理过程来源
from Nizamani import MNB_model, NN_model
from Umer import SVM_model
# from Umer import SVM_GPU

def read_data(mycursor):
    mycursor.execute("SELECT id, summary, description, resolution, preprocessed12, preprocessed12345, sentiment, preprocessed_summary, preprocessed_description from enhancement where resolution is not null and status!='NEW' and creation_time between '1997-09-10' and '2016-07-13'")
    myresult = mycursor.fetchall()
    request = myresult[:]
    # print(len(request))

    id, summary, description, resolution, text, preprocessed12, preprocessed12345, sentiment, preprocessed_summary, preprocessed_description = [], [], [], [], [], [], [], [], [], []
    dict = {"FIXED": 1, "INVALID": 0, "DUPLICATE": 0, "WONTFIX": 0, "INCOMPLETE": 0, "WORKSFORME": 0, "EXPIRED": 0, "MOVED": 0, "INACTIVE": 0}
    dict_senti = {"negative": 0, "positive": 1, "neutral": 2}
    for i, tmp in enumerate(tqdm(request)):
        id.append(tmp[0])
        summary.append(tmp[1].replace("\n", " "))
        description.append(tmp[2].replace("\n", " "))
        resolution.append(dict[tmp[3]])
        text.append(tmp[1].replace("\n", " ") + " " + tmp[2].replace("\n", " "))
        preprocessed12.append(tmp[4].replace("\n", " "))
        preprocessed12345.append(tmp[5].replace("\n", " "))
        sentiment.append(dict_senti[tmp[6]])
        preprocessed_summary.append(tmp[7].replace("\n", " "))
        preprocessed_description.append(tmp[8].replace("\n", " "))
    return id, summary, description, resolution, text, preprocessed12, preprocessed12345, sentiment, preprocessed_summary, preprocessed_description

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

def replace(word, pos=None):
    """" Creates a set of all antonyms for the word and if there is only one antonym, it returns it """
    antonyms = set()
    for syn in nltk.corpus.wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name())
    if len(antonyms) == 1:
        return antonyms.pop()
    else:
        return None

import re
def remove_urls (requests, id, db):
    for index, text in enumerate(tqdm(requests[:])):
        if "http" in text or "https" in text:
            mycursor = db.cursor()
            print(text)
            text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
            print(text)
            try:
                mycursor.execute("UPDATE enhancement SET preprocessed_removeurl='" + escape_string(text) + "' WHERE id=" + str(id[index]))
                db.commit()
            except Exception as e:
                db.rollback()
                print(e)
                print("更新失败1")
                continue
    return(text)

def preprocessing_(requests, id, db):
    # url = "https://twinword-lemmatizer1.p.rapidapi.com/extract/"
    # querystring = {"text": "The frogs hopped from rock to rock.", "flag": "VALIDTOKENSONLYLOWERCASED"}
    # headers = {
    #     'x-rapidapi-key': "2e6cc0feb0msh66dcebed4ca4351p11bde3jsn55e1cb63dc89",
    #     'x-rapidapi-host': "twinword-lemmatizer1.p.rapidapi.com"
    # }
    # response = requests.request("GET", url, headers=headers, params=querystring)
    # print(response.text)


    spell = SpellChecker()
    result = []
    r4 = "\\【.*?<>】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]'"
    for index, text in enumerate(tqdm(requests[:])):
        text = re.sub(r4, ' ', text)  # 正则表达式去掉符号
        mycursor = db.cursor()

        tmp_word_list = nltk.word_tokenize(text) #先tokenize
        interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']  # 定义标点符号列表
        tmp_word_list = [word for word in tmp_word_list if word not in interpunctuations]

        #2 拼写检查
        word_list_spell = []
        for word in tmp_word_list:
            misspelled_words = spell.unknown(word.split())
            if word in misspelled_words:
                word_list_spell.append(spell.correction(word))
            else:
                word_list_spell.append(word)

        # 否定词替换
        word_list_negation = []
        i, l = 0, len(word_list_spell)
        while i < l:
            word = word_list_spell[i]
            if word == 'not' and i + 1 < l:
                ant = replace(word_list_spell[i + 1])
                if ant:
                    word_list_negation.append(ant)
                    i += 2
                    continue
            word_list_negation.append(word)
            i += 1

        # # 存入数据库
        # preprocess_text12 = " ".join(word_list_negation)
        # print(preprocess_text12)
        # try:
        #     mycursor.execute("UPDATE enhancement SET preprocessed12='" + pymysql.escape_string(preprocess_text12) + "' WHERE id=" + str(id[index]))
        #     db.commit()
        # except Exception as e:
        #     db.rollback()
        #     print(e)
        #     print("更新失败1")
        #     continue
        # 3 移除停用词
        stops_list = set(nltk.corpus.stopwords.words('english'))
        word_list_stopwords = []
        for word in word_list_negation:
            if word not in stops_list:
                word_list_stopwords.append(word)

        # 4 inflection and lemmatization
        # https://www.cnblogs.com/jclian91/p/9898511.html结合词性可以更好地还原词形
        tagged_sent = nltk.pos_tag(word_list_stopwords)  # 获取单词词性
        wnl = nltk.WordNetLemmatizer()
        word_list_lemmas = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            word_list_lemmas.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原

        #5 转为小写
        word_list_lemmas = [w.lower() for w in word_list_lemmas]
        text_tmp = " ".join(word_list_lemmas)
        # print(text)
        result.append(text_tmp)
        print(text_tmp)
        # 存入数据库
        try:
            mycursor.execute("UPDATE enhancement SET preprocessed_description='" + pymysql.converters.escape_string(text_tmp) + "' WHERE id=" + str(id[index]))
            db.commit()
        except Exception as e:
            db.rollback()
            print(e)
            print("更新失败2")
            continue
        mycursor.close()
    db.close()
    return result

def VSM(param, x_train, x_test):
    if(param=="TF"):
        vectorizer = CountVectorizer(max_features=20000) #文本中的词语转换成词频矩阵，矩阵元素a[i][j]代表j类词在文本i中的词频
        vectorizer.fit(x_train)
        x_train = vectorizer.transform(x_train).toarray()
        x_test = vectorizer.transform(x_test).toarray()

    if(param=="TFIDF"):
        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(2, 5), encoding='utf-8')
        vectorizer.fit(x_train)
        x_train = vectorizer.transform(x_train).toarray()
        x_test = vectorizer.transform(x_test).toarray()

    return x_train, x_test

def train(model, x_train, x_test, y_train, y_test, confusion_matrix_file):
    if model=="MNB":
        model_mnb = MNB_model(x_train, y_train)
        y_predict = model_mnb.predict(x_test)
        score = model_mnb.score(x_test, y_test)
        # print('MNB训练集分数：', model_mnb.score(x_train, y_train), '测试集分数', score)
        precision, recall, f = evaluate(y_test, y_predict, confusion_matrix_file)
        
    # if model == "SVM-GPU":
    #     model_svm_gpu = SVM_GPU(x_train, y_train)
    #     y_predict = model_svm_gpu.predict(x_test)
    #     score = model_svm_gpu.score(x_test, y_test)
    #     precision, recall, f = evaluate(y_test, y_predict)

    if model == "SVM":
        model_svm = SVM_model(x_train, y_train)
        y_predict = model_svm.predict(x_test)
        score = model_svm.score(x_test, y_test)
        # print('SVM训练集分数：', model_svm.score(x_train, y_train), '测试集分数',score)
        precision, recall, f = evaluate(y_test, y_predict, confusion_matrix_file)

    if model=="NN":
        model_nn = NN_model(x_train, y_train)
        y_predict = model_nn.predict(x_test)
        score = model_nn.score(x_test, y_test)
        # print('NN训练集分数：', model_nn.score(x_train, y_train), '测试集分数', score)
        precision, recall, f = evaluate(y_test, y_predict, confusion_matrix_file)

    return precision, recall, f, score

# 用户评估precision recall f
def evaluate(y_true, y_pred, confusion_matrix_file):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    confusion_matrix = confusion_matrix(y_true, y_pred)
    p_r_f = precision_recall_fscore_support(y_true, y_pred)
    confusion_matrix_file.write(str(confusion_matrix) + '\n') 
    confusion_matrix_file.write("precision, recall, fscore:\n{}\n".format(p_r_f))  
    # print("confusion_matrix: ", confusion_matrix)

    # 计算第i类的precision, recall and F
    diag = np.diag(confusion_matrix)  # 取对角线上的值
    raw_sum = np.sum(confusion_matrix, axis=1)  # 计算每一行的和
    each_recall = np.nan_to_num(truediv(diag, raw_sum))
    # print("recall: ", each_acc)

    column_sum = np.sum(confusion_matrix, axis=0)  # 计算每一列的和
    each_precision = np.nan_to_num(truediv(diag, column_sum))
    # print("precision: ", each_precision)

    F = []
    i = 0
    while i < len(each_precision):
        F.append(2 * each_precision[i] * each_recall[i] / (each_precision[i] + each_recall[i]))
        i = i + 1

    # print(each_precision)
    # print(each_recall)
    # print(F)
    return each_precision, each_recall, F

