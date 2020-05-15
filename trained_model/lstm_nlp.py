
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer 
from nltk.tokenize import word_tokenize 
stemmer = PorterStemmer() 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
lemmatizer = WordNetLemmatizer() 

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re

from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential
# import the necessary libraries 
import nltk 
import string 
import re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

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
  
# Function to convert list to string  
def list_to_string(s):  
    str1 = " "   
    return (str1.join(s))




#read dataset/home/devyani/Desktop/dl_project_3
data_dir = '/home/devyani/Desktop/dl_project_3/data/'
# Saved in the d2l package for later use
def read_snli(file_name):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\(', '', s)
        s = re.sub('\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2, '-':3}
    #file_name = (data_dir + 'snli_1.0_train.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels

data = read_snli(data_dir + 'snli_1.0_train.txt')
#data_test=read_snli(data_dir + 'snli_1.0_test.txt')



def preprocessing(data):
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

	#onehot_encoder = OneHotEncoder(sparse=False)
	y=data[2]
	y=np.array(y)
	onehot_encoder = OneHotEncoder(sparse=False)
	y = y.reshape(len(y), 1)
	y= onehot_encoder.fit_transform(y)
	print(y)

	print(texts[0],y[0])
	return texts,y

texts,y=preprocessing(data)
#texts_test,y_test=preprocessing(data_test)

MAX_NB_WORDS = 100000   # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 60 # max length of each entry (sentence), including padding
VALIDATION_SPLIT = 0.2   # data for validation (not used in training)
EMBEDDING_DIM = 50      # embedding dimensions for word vectors (word2vec/GloVe)
GLOVE_DIR = "/home/devyani/Desktop/dl_project_3/glove.6B/glove.6B."+str(EMBEDDING_DIM)+"d.txt"

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Vocabulary size:', len(word_index))


data = pad_sequences(sequences, padding = 'post', maxlen = MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y.shape)


num_validation_samples = int(VALIDATION_SPLIT*data.shape[0])
x_train = data[: -num_validation_samples]
y_train = y[: -num_validation_samples]
x_val = data[-num_validation_samples: ]
y_val = y[-num_validation_samples: ]
print('Number of entries in each category:')
print('training: ', y_train.sum(axis=0))
print('validation: ', y_val.sum(axis=0))

print('Tokenized sentences: \n', data[10])
print('One hot label: \n', y[10])



embeddings_index = {}
f = open(GLOVE_DIR)
print('Loading GloVe from:', GLOVE_DIR,'...', end='')
for line in f:
    values = line.split()
    word = values[0]
    embeddings_index[word] = np.asarray(values[1:], dtype='float32')
f.close()
print("Done.\n Proceeding with Embedding Matrix...", end="")

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print(" Completed!")


model = Sequential()
model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
model.add(Embedding(len(word_index) + 1,
 EMBEDDING_DIM,
 weights = [embedding_matrix],
 input_length = MAX_SEQUENCE_LENGTH,
 trainable=False,
 name = 'embeddings'))
model.add(Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer')))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.1))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(4, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy',
             optimizer='adam',
             metrics = ['accuracy'])


print(x_train.shape)
print(y_train.shape)
print('Training progress:')
history=model.fit(x_train, y_train, epochs = 10, batch_size=32, validation_data=(x_val, y_val))
model.save('sentiment_analysis.h5')

fig=plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
fig.savefig('train_val_acc.png')
plt.show()




