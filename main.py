# -*- coding: utf-8 -*-

from sklearn.preprocessing import OneHotEncoder
import nltk 
import string 
import re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer 
from nltk.tokenize import word_tokenize 
stemmer = PorterStemmer() 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
lemmatizer = WordNetLemmatizer()

from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
#to lower case
def text_lowercase(text): 
    return text.lower() 

# remove punctuation 
def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator) 

# remove whitespace from text 
def remove_whitespace(text): 
    return  " ".join(text.split()) 
 
# remove stopwords function 
def remove_stopwords(text): 
    stop_words = set(stopwords.words("english")) 
    word_tokens = word_tokenize(text) 
    filtered_text = [word for word in word_tokens if word not in stop_words] 
    return filtered_text 
   
# stem words in the list of tokenised words 
def stem_words(text): 
    word_tokens = word_tokenize(text) 
    stems = [stemmer.stem(word) for word in word_tokens] 
    return stems 
  
# lemmatize string 
def lemmatize_word(text): 
    word_tokens = word_tokenize(text) 
    # provide context i.e. part-of-speech 
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens] 
    return lemmas 
    
# Function to convert list_to_string  
def list_to_string(s):  
    # initialize an empty string 
    str1 = " " 
    # return string   
    return (str1.join(s))

#read dataset
data_dir = 'data/'
# Saved in the d2l package for later use
def read_snli(data_dir):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\(', '', s)
        s = re.sub('\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2, '-':3}
    file_name = (data_dir + 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels

data = read_snli(data_dir)
n=np.array(data).shape[1]
i=0
texts=[]
for x0, x1, y1 in zip(data[0][:n], data[1][:n],data[2][:n]):
    x0=text_lowercase(x0) 
    x0=remove_punctuation(x0)
    x0=remove_whitespace(x0)
    #x0=stem_words(x0)
    #x0=list_to_string(x0)
    x0=lemmatize_word(x0)
    x0=list_to_string(x0)
    x0=remove_stopwords(x0)
    x0=list_to_string(x0)
    x1=text_lowercase(x1)
    x1=remove_punctuation(x1)
    x1=remove_whitespace(x1)
    #x1=stem_words(x1)
    #x1=list_to_string(x1)
    x1=lemmatize_word(x1)
    x1=list_to_string(x1)
    x1=remove_stopwords(x1)
    x1=list_to_string(x1)
    #print('premise:', x0)
    #print('hypothesis:', x1)
    #print('label:', y)
    texts.append(x0+' '+x1)


y=data[2]
y=np.array(y)
ytest=y
onehot_encoder = OneHotEncoder(sparse=False)
y = y.reshape(len(y), 1)
y= onehot_encoder.fit_transform(y)
print(y)
print(texts[0],y[0])


MAX_NB_WORDS = 100000   # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 60 # max length of each entry (sentence), including padding
#VALIDATION_SPLIT = 0  # data for validation (not used in training)
EMBEDDING_DIM = 50      # embedding dimensions for word vectors (word2vec/GloVe)
GLOVE_DIR = "glove.6B/glove.6B."+str(EMBEDDING_DIM)+"d.txt"

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Vocabulary size:', len(word_index))


data = pad_sequences(sequences, padding = 'post', maxlen = MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y.shape)


#prepare data for lstm-model
x_test = data
y_test = y
#print(x_test.shape)
#print(y_test.shape)

#load lstm model

lstm_model=load_model('lstm_trained_model.h5')
lstm_model.summary()
y_pred1 =lstm_model.predict_classes(x_test)

#load regression model

filename = 'regression_trained_model.sav'
regression_loaded_model = pickle.load(open(filename, 'rb'))

#load tfidf vectors

tf1=pickle.load(open("tfidf_feature.pkl", 'rb'))
tfidf_test_x=tf1.transform(texts)
scores = cross_val_score(regression_loaded_model, tfidf_test_x,ytest, cv=5)

#testing on regression model
y_reg=regression_loaded_model.predict(tfidf_test_x)
acc = scores.mean()
print ("Accuracy: %0.2f percent" % (acc *100))

#testing on lstm model
score = lstm_model.evaluate(x_test,y_test)
print(score)
output1=[]
output2=[]


output1.append('gt_label,pred_label')
output2.append('gt_label,pred_label')

for i in range(len(x_test)):
	if(ytest[i]==0):
		a='entailment'
	elif(ytest[i]==1):
		a='contradiction'
	elif(ytest[i]==2):
		a='neutral'
	else:
		a='-'

	if(y_reg[i]=='0'):
		c='entailment'
	elif(y_reg[i]=='1'):
		c='contradiction'
	elif(y_reg[i]=='2'):
		c='neutral'
	else:
		c='-'

	if(y_pred1[i]==0):
		b='entailment'
	elif(y_pred1[i]==1):
		b='contradiction'
	elif(y_pred1[i]==2):
		b='neutral'
	else:
		b='-'
	output1.append(str(a)+','+str(b))
	output2.append(str(a)+','+str(c))
	#output2.append(str(y_test[i])+','+str(y_pred2[i]))
np.savetxt('deep_model.txt', output1, delimiter =" ", fmt="%s") 
np.savetxt('tfidf.txt', output2, delimiter =" ", fmt="%s")

