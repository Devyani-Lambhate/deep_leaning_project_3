from nltk.stem.porter import PorterStemmer 
from nltk.tokenize import word_tokenize 
stemmer = PorterStemmer() 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
lemmatizer = WordNetLemmatizer() 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import pickle
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
   
def list_to_string(s):  
    str1 = " "   
    return (str1.join(s))





#read dataset
data_dir = '/home/devyani/Desktop/dl_project_3/data/'
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
    file_name = (data_dir + 'snli_1.0_train.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels

data = read_snli(data_dir)
n=np.array(data).shape[1]
texts=[]
for x0, x1, y1 in zip(data[0][:n], data[1][:n],data[2][:n]):
    #print(x0)
    x0=text_lowercase(x0) 
    x0=remove_punctuation(x0)
    x0=remove_whitespace(x0)
    #x0=stem_words(x0)
    #x0=list_to_string(x0)
    x0=lemmatize_word(x0)
    x0=list_to_string(x0)
    x0=remove_stopwords(x0)
    x0=list_to_string(x0)
    #print(x0)
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
    lista=x0+' '+x1
    texts.append(lista)
#print(train_data[2][0:100])
data=np.array(data)
#print(data.shape)
size=np.array(data).shape[1]
datanew=np.zeros((2,size))
texts=np.array(texts)
print(texts.shape)



#for i in range(np.array(train_data).shape[1]):
 # train_data[0][i]=train_data[0][i]+train_data[1][i]

texts=texts.T
y=data[2]
y=np.array(y)
print(y.shape)



train_x, test_x, train_y, test_y = train_test_split(texts,y)

train_x=np.array(train_x)
train_y=np.array(train_y)
test_x=np.array(test_x)                                                   
test_y=np.array(test_y)



print(train_x.shape)

print(test_x.shape)
print(test_y.shape)

train_size=train_x.shape[0]
test_size=test_x.shape[0]
print(train_x[0],y[0])

print(y.shape)

vectorizer = TfidfVectorizer()
tfidf_train_x = vectorizer.fit_transform(train_x)
pickle.dump(vectorizer,open("tfidf_feature.pkl","wb"))
print(tfidf_train_x.shape)
size_of_tfidfvec=tfidf_train_x.shape[1]

print(tfidf_train_x.shape)
print(train_y.shape)

classifier = LogisticRegression(max_iter=1000)#,verbose=1,solver='liblinear',random_state=0, C=5, penalty='l2')
classifier.fit(tfidf_train_x, train_y)

tfidf_test_x = vectorizer.transform(test_x)

scores = cross_val_score(classifier, tfidf_test_x, test_y, cv=5)
acc = scores.mean()
print ("Accuracy: %0.2f percent" % (acc *100))

import pickle
filename = 'trained_regression_model.sav'
pickle.dump(classifier, open(filename, 'wb'))







