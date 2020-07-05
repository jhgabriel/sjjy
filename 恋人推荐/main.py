import multiprocessing
import pandas as pd
import numpy as np
import jieba
import pickle
np.random.seed(1337)  # For Reproducibility


import yaml
import sys
import os
import math
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


from gensim.models import Doc2Vec
from gensim.corpora.dictionary import Dictionary
from gensim.models import Word2Vec

import keras
import keras.utils
from keras import utils as np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml



# 切换到当前脚本所在目录,并创建模型目录备用
os.chdir(os.path.dirname(os.path.abspath(__file__)));
if not os.path.exists("./model"):
    os.makedirs("./model");

def get_file_path_in_model(file_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),"model",file_name)

sys.setrecursionlimit(1000000)#递归的最大深度
#%% set parameters:
vocab_dim = 100
n_iterations = 1  # ideally more..
n_exposures = 10 #词频数少于10的截断
window_size = 7
batch_size = 32
n_epoch = 4
input_length = 188 #LSTM输入 注意与下长度保持一致
maxlen = 188#统一句长
cpu_count = multiprocessing.cpu_count()


#%%加载训练文件
def load_trainning_file():
    df = pd.read_csv('../世纪佳缘_去重_UserInfo_男_label.csv',nrows=500,
                           names=['age', 'shortnote','judge'])
    df['y'] = 0 # 新增一列分值列
    for index,r in df.iterrows():
        #print("%d: %s" % (index,r['shortnote']))
        if(pd.isna( r['shortnote'])):
            print("%d: %s" % (index,r['shortnote']))
            df['shortnote'][index]= ' '
            print("dddddddddddddddddd")

        if r['judge'] == '平凡':
            df['y'][index]= 0
        elif r['judge'] == '优秀':
            df['y'][index]= 1
        elif r['judge'] == '卓越':
            df['y'][index]= 2
        else:
            print("bad data")
            quit();

    
    combined= df['shortnote'].to_numpy()
    y = df['y'].to_numpy()#添加标注
    return combined,y


#%%对句子进行分词，并去掉换行符
def tokenizer(documents):

    def op(document):
        ret = ""
        #print(type(document))
        #print(document)
        ret = jieba.lcut(document.replace('\n', ''))
        return ret
    text_list = [op(document) for document in documents]
    return text_list


#%%创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量

        def parse_dataset(combined):
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#前方补0 为了进入LSTM的长度统一
        #每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print('No data provided...')


#%%创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):
    
    model = Word2Vec(size=vocab_dim,#特征向量维度
                     min_count=n_exposures,#可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
                     window=window_size,#窗口大小，表示当前词与预测词在一个句子中的最大距离是多少
                     workers=cpu_count,#用于控制训练的并行数
                     iter=n_iterations)
    model.build_vocab(combined)#创建词汇表， 用来将 string token 转成 index
    model.train(combined,total_examples=model.corpus_count,epochs=10)
    model.save('./model/Word2vec_model.pkl')#保存训练好的模型
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined#word_vectors字典类型{word:vec}

#%%最终的数据准备
def get_data(index_dict,word_vectors,combined,y):

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
        
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    y_train = keras.utils.to_categorical(y_train,num_classes=3) 
    y_test = keras.utils.to_categorical(y_test,num_classes=3)
    # print x_train.shape,y_train.shape
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test



#######################################
# 模型构建
#######################################

#%%定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever #堆叠
    #嵌入层将正整数（下标）转换为具有固定大小的向量
    model.add(Embedding(output_dim=vocab_dim,#词向量的维度
                        input_dim=n_symbols,#字典(词汇表)长度
                        mask_zero=True,#确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length#当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断。
    #输入数据的形状为188个时间长度（句子长度），每一个时间点下的样本数据特征值维度（词向量长度）是100。
    model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
    #输出的数据，时间维度仍然是188，每一个时间点下的样本数据特征值维度是50
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax')) # Dense=>全连接层,输出维度=3
    model.add(Activation('softmax'))

    print('Compiling the Model...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch,verbose=1, validation_data=(x_test, y_test))

    print("Evaluate...")
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('./model/lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('./model/lstm.h5')
    print('Test score:', score)


#%%
#######################################
# 训练模型
#######################################
#
def train_model():
    print('======== Loading Data... ========')
    combined,y=load_trainning_file()
    print(len(combined),len(y))
    print('======== Tokenising... ========')
    combined = tokenizer(combined)
    print('======== Training a Word2vec model... ========')
    index_dict, word_vectors,combined=word2vec_train(combined)
    print('======== Setting up Arrays for Keras Embedding Layer... ========')
    n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
    print(x_train.shape,y_train.shape)
    train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)


#%%
#######################################
# 测试模型
#######################################
def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('./model/Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined

def lstm_predict(string):
    print('loading model......')
    with open('./model/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('./model/lstm.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    data=input_transform(string)
    data.reshape(1,-1)
    #print data
    result=model.predict_classes(data)
    if result[0]==0:
        print(string,' 平凡')
    elif result[0]==1:
        print(string,' 优秀')
    elif result[0]==2:
        print(string,' 卓越')
    else:
        result[0] = 0
        print(string,' 未知')

    return result[0]

def test_model():
    string='我是一个孝顺,顾家的人 ，爱好广泛的我，喜欢电影'
    lstm_predict(string)  


base_education_score = {"未知":20,"高中中专及以下":30,"大专":50,"本科":90,"硕士":100,"双学士":100,"博士":110}
base_shortnote_score = {"平凡":30,"优秀":50,"卓越":100}
base_shortnoteindex_score = [30,50,100]
base_age_section = [[22,30],[24,32]]
base_height_section = [[160,175],[168,185]]

def score_item(model,sex,age,education,height,shortnote):
    def get_age_score(sex,age):
        score = 30
        sec = base_age_section[1]
        if(sex == "女"):
            sec = base_age_section[0]

        if(age < sec[0]):
            score -= abs(sec[0]-age);

        if(age > sec[1]):
            score -= abs(age-sec[1]);

        if(score<0):
            score = 0; 
              
        return score;

    def get_height_score(sex,height):
        score = 30
        sec = base_height_section[1]
        if(sex == "女"):
            sec = base_height_section[0]

        if(height < sec[0]):
            score -= abs(sec[0]-height);

        if(height > sec[1]):
            score -= abs(height-sec[1]);

        if(score<0):
            score = 0; 
              
        return score;

    def get_education_score(sex,education):
        return base_education_score[education]*1.0;

    def get_shortnote_score(sex,shortnote):
        if(pd.isna(shortnote)):
            return 0;

        data=input_transform(shortnote)
        data.reshape(1,-1)
        class_index=model.predict_classes(data)
        return base_shortnoteindex_score[class_index[0]]

    return (get_age_score(sex,age) + \
            get_education_score(sex,education) + \
            get_height_score(sex,height) + \
            get_shortnote_score(sex,shortnote) ,
            get_age_score(sex,age),
            get_education_score(sex,education),
            get_height_score(sex,height),
            get_shortnote_score(sex,shortnote) 
    );


def create_score_file(file_path):
    bn = os.path.basename(file_path)
    name = os.path.splitext(bn)[0]
    ext = os.path.splitext(bn)[1]

    df = pd.read_csv(file_path,#nrows=100,
                           names=['uid', 'nickname', 'sex', 'age', 'work_location', 'height', 'education',
                                  'matchCondition', 'marriage', 'income', 'shortnote', 'image'])
    df['score'] = 0 # 新增一列分值列

    df['age_score'] = 0 # 新增一列分值列
    df['eduction_score'] = 0 # 新增一列分值列
    df['height_score'] = 0 # 新增一列分值列
    df['shortnote_score'] = 0 # 新增一列分值列
    with open('./model/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('./model/lstm.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])


    for index,r in df.iterrows():
        #df['score'][index]= score_item(model,r['sex'],r['age'],r['education'],r['height'],r['shortnote'])[0]
        df['score'][index],df['age_score'][index],df['eduction_score'][index],df['height_score'][index],df['shortnote_score'][index]= score_item(model,r['sex'],r['age'],r['education'],r['height'],r['shortnote'])

        print("%d:(%.2f) %s" % (index,df['score'][index],r['shortnote']))

    output_path = os.path.join('.','model',name+"_score.csv");
    df.to_csv(output_path, encoding='utf-8-sig', mode='w', index=False, sep=',', header=False)

def score_all():
    create_score_file('../世纪佳缘_去重_UserInfo_男.csv')
    create_score_file('../世纪佳缘_去重_UserInfo_女.csv')



def match_lover(sex,age,education,height,shortnote):
    with open('./model/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
        model = model_from_yaml(yaml_string)
        model.load_weights('./model/lstm.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        score = score_item(model,sex,age,education,height,shortnote)[0]

        file_path= get_file_path_in_model('世纪佳缘_去重_UserInfo_女_score.csv')
        if(sex=="女"):
            file_path= get_file_path_in_model('世纪佳缘_去重_UserInfo_男_score.csv')
        print(file_path)
        df = pd.read_csv(file_path,#nrows=100,
                           names=['uid', 'nickname', 'sex', 'age', 'work_location', 'height', 'education',
                                  'matchCondition', 'marriage', 'income', 'shortnote', 'image','score','age_score','education_score','height_score','shortnote_score'])

        count = 0;
        for index,r in df.iterrows(): 
            if(r['score']==score):
                #print("%d:(%.2f) %s" % (index,df['score'][index],r['shortnote']))  
                print("%s,%d,%s,%d,%s" % (r['sex'],r['age'],r['education'],r['height'],r['shortnote']))  
                count = count +1
                if(count >=10):
                    break



#%%测试
if __name__=='__main__':
    
    #加载训练数据，训练模型
    #train_model() # 打开本行注释以重新训练模型
    #test_model() # 打开本行注释以简单测试几个用例

    #为真实数据生成 score 文件
    #score_all()

    #根据score数据做推荐，输入自身数据，给出若干匹配对象
    match_lover('男',28,'本科',177,'我本善良')




# %%
