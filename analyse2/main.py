import multiprocessing
import pandas as pd
import numpy as np
import jieba
import pickle
np.random.seed(1337)  # For Reproducibility


import yaml
import sys
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
def loadfile():
    df = pd.read_csv('./jiayuan/marked.csv',
                           names=['uid', 'nickname', 'sex', 'age', 'work_location', 'height', 'education',
                                  'matchCondition', 'marriage', 'income', 'shortnote', 'image','judge'])
    df['y'] = 0
    for index,r in df.iterrows():
        #df['nickname'][index]= r['education']+';'+r['marriage']+';'+r['shortnote']
        df['nickname'][index]= r['education']+';'+str(r['age'])
        if r['judge'] == '一般':
            df['y'][index]= -1
        elif r['judge'] == '良好':
            df['y'][index]= 0
        elif r['judge'] == '优秀':
            df['y'][index]= 1
        else:
            print("bad data")
            quit();


    print(type(df))
    combined= df['nickname'].to_numpy()
    y = df['y'].to_numpy()#添加标注
    print(type(combined))
    print(type(y))
    return combined,y

#%%对句子进行分词，并去掉换行符
def tokenizer(text):
    #text = [print(type(document)) for document in text]
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    print(text)
    return text


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
    model.save('./jiayuan/Word2vec_model.pkl')#保存训练好的模型
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
    with open('./jiayuan/lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('./jiayuan/lstm.h5')
    print('Test score:', score)


#%%
#######################################
# 训练模型
#######################################
#
def train_model():
    print('Loading Data...')
    combined,y=loadfile()
    print(len(combined),len(y))
    print('Tokenising...')
    combined = tokenizer(combined)
    print('Training a Word2vec model...')
    index_dict, word_vectors,combined=word2vec_train(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
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
    model=Word2Vec.load('./jiayuan/Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined

def lstm_predict(string):
    print('loading model......')
    with open('./jiayuan/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('./jiayuan/lstm.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    data=input_transform(string)
    data.reshape(1,-1)
    #print data
    result=model.predict_classes(data)
    if result[0]==1:
        print(string,' 优秀')
    elif result[0]==0:
        print(string,' 良好')
    else:
        print(string,' 一般')




def test_model():
    string='本科;26'
    lstm_predict(string)  
    string='硕士;31'
    lstm_predict(string) 

#%%测试
if __name__=='__main__':
    #train_model()
    test_model()


# %%
