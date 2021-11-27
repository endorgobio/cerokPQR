# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 18:22:39 2021

@author: pmayaduque
"""

import pandas as pd
import wget
import numpy as np
import re
import seaborn as sns


# Preprocessing and modelin
# ==============================================================================
import string
import scipy 
import spacy
from spacy.lang.es import Spanish
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Save de model
import pickle


# 1. Read data
data_url = 'https://docs.google.com/uc?export=download&id=1tIeykgOX81-jwrc-vYpHI-PLGTg6q5kl'
dfPQR = pd.read_csv(wget.download(data_url))

dfPQR.rename(columns={'Contenido de la PQRS' : 'Contenido',
                      'Riesgo de vida' : 'Riesgo_vida',
                      'Canal Origen PQR' : 'Canal_PQR',
                      'Edad Solicitante' : 'Edad',
                      'Genero Solicitante' : 'Genero'
                      },
             inplace=True) 

# Correct those that contains 'riesgo de vida' but were not labaled so
def explicit_riesgo(content, actual_riesgo):
    if actual_riesgo == 'SI':
        return actual_riesgo
    elif 'riesgo de vida' in str(content):#.contains('riesgo de vida'):
        return 'SI'
    else:
        return actual_riesgo
dfPQR['Riesgo_vida'] = dfPQR.apply(lambda x: explicit_riesgo(
    x['Contenido'], x['Riesgo_vida']), axis=1)

# Remove all rows with NA in column Contenido
dfPQR = dfPQR.dropna(subset=['Contenido'])

# Filter NA in in column 'Riesgo de vida'  
dfPQR = dfPQR.dropna(subset=['Riesgo_vida'])


# Convert contenido to lowercase
dfPQR["Contenido"] = dfPQR["Contenido"].str.lower()

# Search for 'riesgo de vida' in the content
#dfPQR['riesgo_vida_expl'] = dfPQR['Contenido'].str.contains('riesgo de vida')



# Applying encoding to the Riesgo_vida column
#dfPQR['Riesgo_vida_encode'] = dfPQR['Riesgo_vida'].factorize()[0] 
dfPQR['Riesgo_vida_encode'] = [1 if x =='SI' else 0 for x in dfPQR['Riesgo_vida']]

#stop_words = list(stopwords.words('spanish'))
# Se añade la stopword: amp, ax, ex
#stop_words.extend(("amp", "xa", "xe"))


# NLP 
nlp = Spanish()
nlp = spacy.load("es_core_news_sm")
stop_words = spacy.lang.es.stop_words.STOP_WORDS
spanish_stopwords = set(stopwords.words('spanish'))
punctuations = string.punctuation


# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = nlp(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    
    # Removing numbers and words with less than 4 letters
    mytokens = [word for word in mytokens if len(word) > 3 ]
    
    # return preprocessed list of tokens
    return mytokens

tfidf_vectorizador = TfidfVectorizer(    
                        tokenizer  = spacy_tokenizer,
                        max_df = 1.0,
                        min_df = 5,
                        stop_words = spanish_stopwords
                    )

weights = {0:1, 1:2.5}
classifier = LogisticRegression(class_weight=weights)

#####################################################
# 1. with only contenido as predictor  
#####################################################

# To run with less data in debug (faster)
dfPQR = dfPQR.sample(1000)
# Get data
X_data = dfPQR['Contenido']
Y_data = dfPQR['Riesgo_vida_encode']

X_train, X_test, y_train, y_test = train_test_split(
    X_data,
    Y_data,
    test_size = 0.2,
    random_state = 42    
)

# Check distributions in train and test sets
value, counts = np.unique(y_train, return_counts=True)
print(dict(zip(value, 100 * counts / sum(counts))))
value, counts = np.unique(y_test, return_counts=True)
print(dict(zip(value, 100 * counts / sum(counts))))

# Fit the vectorizer
tfidf_vectorizador.fit(X_train)
# transform X_train and X_test
tfidf_train = tfidf_vectorizador.transform(X_train)
tfidf_test  = tfidf_vectorizador.transform(X_test)

# Fit the classifier
classifier.fit(X=tfidf_train, y= y_train)

# Get predictions

predicted = classifier.predict(X=tfidf_test)
# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))


#####################################################
# 2. with only contenido as predictor using pipelines 
#####################################################

# To run with less data in debug (faster)
dfPQR = dfPQR.sample(1000)
# Get data 
X_data = dfPQR['Contenido']
Y_data = dfPQR['Riesgo_vida_encode']

X_train, X_test, y_train, y_test = train_test_split(
    X_data,
    Y_data,
    test_size = 0.2,
    random_state = 42    
)

# Check distributions in train and test sets
value, counts = np.unique(y_train, return_counts=True)
print(dict(zip(value, 100 * counts / sum(counts))))
value, counts = np.unique(y_test, return_counts=True)
print(dict(zip(value, 100 * counts / sum(counts))))


# Create pipeline using tfidf_vectorizador
pipe = Pipeline([('vectorizer', tfidf_vectorizador),
                 ('classifier', classifier)])


# model generation
pipe.fit(X_train,y_train)

# Predicting with a test dataset
predicted = pipe.predict(X_test)

# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))

#####################################################
# 3. with contenido and extra columns as predictors 
#####################################################

# To run with less data in debug (faster)
# dfPQR = dfPQR.sample(1000)
# Get data 
X_data = dfPQR[['Contenido', 'Edad']]
Y_data = dfPQR['Riesgo_vida_encode']

X_train, X_test, y_train, y_test = train_test_split(
    X_data,
    Y_data,
    test_size = 0.2,
    random_state = 42    
)

# Check distributions in train and test sets
value, counts = np.unique(y_train, return_counts=True)
print(dict(zip(value, 100 * counts / sum(counts))))
value, counts = np.unique(y_test, return_counts=True)
print(dict(zip(value, 100 * counts / sum(counts))))


# Fit the vectorizer
tfidf_vectorizador.fit(X_train['Contenido'])
# transform X_train and X_test
tfidf_train = tfidf_vectorizador.transform(X_train['Contenido'])
tfidf_test  = tfidf_vectorizador.transform(X_test['Contenido'])

numerical_trasnsf = ColumnTransformer(
    [('scaler', MinMaxScaler(), ['Edad'])],
    remainder='passthrough') 
edad_scaled_train = numerical_trasnsf.fit_transform(X_train['Edad'].to_frame())
edad_scaled_test = numerical_trasnsf.fit_transform(X_test['Edad'].to_frame())


tfidf_train_plus = scipy.sparse.hstack([tfidf_train, edad_scaled_train])
tfidf_test_plus = scipy.sparse.hstack([tfidf_test, edad_scaled_test])

# Fit the classifier
classifier.fit(X=tfidf_train_plus, y= y_train)

# Get predictions

predicted = classifier.predict(X=tfidf_test_plus)

# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))


# Save the models
# create an iterator object with write permission - model.pkl
with open('vectorizer_extraFeatures.pkl', 'wb') as files:
    pickle.dump(tfidf_vectorizador, files)

# create an iterator object with write permission - model.pkl
with open('lg_extraFeatures.pkl', 'wb') as files:
    pickle.dump(classifier, files)


    
    
# Test models
# load saved model
with open('vectorizer_extraFeatures.pkl' , 'rb') as f:
    vectorizer_load = pickle.load(f)
    
with open('lg_extraFeatures.pkl' , 'rb') as f:
    classifier_load = pickle.load(f)
  

tfidf_test2  = vectorizer_load.transform(X_test['Contenido'])
edad_scaled_test = numerical_trasnsf.fit_transform(X_test['Edad'].to_frame())
tfidf_test_plus = scipy.sparse.hstack([tfidf_test, edad_scaled_test])
predicted = classifier.predict(X=tfidf_test_plus)

print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))

#####################################################
# 4. with contenido and extra columns as predictors using pipeline 
#####################################################


column_trans = ColumnTransformer(
    [('scaler', StandardScaler(), ["Edad"])],
    remainder='passthrough') 
data = column_trans.fit_transform(X_train)

new_feature  =  data[:,0]
tfidf_train_plus = scipy.sparse.hstack([tfidf_train, new_feature])
new_feature = dfPQR[['Edad']].sample(tfidf_test.shape[0])
new_feature['Edad']  = np.random.rand()
tfidf_test_plus = scipy.sparse.hstack([tfidf_test, new_feature])




class PqrTextProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.TfidfVectorizer = Pipeline(steps=[
        ('tfidf', tfidf_vectorizador)    ])
       
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
       
        return  self.TfidfVectorizer.fit_transform(X.squeeze()).toarray()
    
    
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('tweet', PqrTextProcessor(), ['Contenido']),
    ('numeric', numeric_transformer, ['Edad'])
])


weights = {0:1, 1:2.5}
classifier = LogisticRegression(class_weight=weights)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])


pipeline.fit(X_train, y_train)

predicted = pipeline.predict(X=X_test)

# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))


feature_text = ['Contenido']
feature_num = ['Edad']
# construct the column transfomer
column_transformer = ColumnTransformer(
    [('tfidf', tfidf_vectorizador, feature_text), 
    #('Scaler', StandardScaler(), feature_num)
    ],
    remainder='passthrough')

data = column_transformer.fit_transform(X_train)

weights = {0:1, 1:2.5}
classifier = LogisticRegression(class_weight=weights)

# fit the model
pipe = Pipeline([
                  ('tfidf', column_transformer),
                  ('classify', classifier)
                ])
pipe.fit(X_train, y_train)

tfidf_vectorizador.fit(X_train)
tfidf_train = tfidf_vectorizador.transform(X_train)
tfidf_test  = tfidf_vectorizador.transform(X_test)





# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))

#####################################################
# 5. Print performance metrics 
#####################################################


confusion_matrix(y_true = y_test, y_pred= predicted)


frame = {'Contenido': X_test, 'y_real': y_test, 'y_hat': predicted }
  
result = pd.DataFrame(frame)

result['Riesgo_vida_textual'] = [1 if 'riesgo de vida' in str(x) else 0 for x in result['Contenido']]
result_grouped = result.groupby(['Riesgo_vida_textual', 'y_real', 'y_hat']).y_hat.agg(
    suma=('count')
)
  

#####################################################
# 6. Save models using pickle 
#####################################################

# save the model to disk
filenameVect = 'vectorizerPQRs.sav'
pickle.dump(pipe['vectorizer'], open(filenameVect, 'wb'))
filenameLR = 'logRegPQRs.sav'
pickle.dump(pipe['classifier'], open(filenameLR, 'wb'))

# Load models from pickle
vectorizer_url = 'https://docs.google.com/uc?export=download&id=1QOVxd0R7UctnUpwlDKldHgUhIPFSt32M'
vectorizerPQRs = wget.download(vectorizer_url, 'vectorizerPQRs.sav')
lr_url = 'https://docs.google.com/uc?export=download&id=1y-pDeJCWM413aijsux4iMVP_bjRcE73X'
lrPQRs = wget.download(lr_url, 'lrPQRs.sav')

loaded_vectorizer = pickle.load(open(vectorizerPQRs, 'rb'))
loaded_logReg = pickle.load(open(lrPQRs, 'rb'))






#########################################################

from spacy.lang.en import English
# Definir el propio tokenizador
def limpiar_tokenizar(texto):
    '''
    Esta función limpia y tokeniza el texto en palabras individuales.
    El orden en el que se va limpiando el texto no es arbitrario.
    El listado de signos de puntuación se ha obtenido de: print(string.punctuation)
    y re.escape(string.punctuation)
    '''
    
    # Se convierte todo el texto a minúsculas
    nuevo_texto = texto.lower()
    # Eliminación de páginas web (palabras que empiezan por "http")
    nuevo_texto = re.sub('http\S+', ' ', nuevo_texto)
    # Eliminación de signos de puntuación
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    nuevo_texto = re.sub(regex , ' ', nuevo_texto)
    # Eliminación de números
    nuevo_texto = re.sub("\d+", ' ', nuevo_texto)
    # Eliminación de espacios en blanco múltiples
    nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)    
    # Crea un objeto de spacy tipo nlp
    doc = nlp(nuevo_texto)
    #words = [t.orth_ for t in doc if not t.is_punct | t.is_stop]
    #lexical_tokens = [t for t in words if len(t) > 3 and t.isalpha()]
    lemmas = [tok.lemma_ for tok in doc if len(tok) > 3]
    # Tokenización por palabras individuales
    
    #nuevo_texto = nuevo_texto.split(sep = ' ')
    # Eliminación de tokens con una longitud < 2
    #nuevo_texto = [token for token in nuevo_texto if len(token) > 1]
    
    return lemmas

nlp = spacy.load('es_core_news_sm')
nlp = spacy.load("es_core_news_md")
spanish_stopwords = set(stopwords.words('spanish'))
nlp = Spanish()
def tokenizer (texto):
    # Se convierte todo el texto a minúsculas
    nuevo_texto = texto.lower()
    # Eliminación de páginas web (palabras que empiezan por "http")
    nuevo_texto = re.sub('http\S+', ' ', nuevo_texto)
    # Eliminación de signos de puntuación
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    nuevo_texto = re.sub(regex , ' ', nuevo_texto)
    # Eliminación de números
    nuevo_texto = re.sub("\d+", ' ', nuevo_texto)
    # Eliminación de espacios en blanco múltiples
    nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
    parsed_phrase = nlp(nuevo_texto)
    #for token in parsed_phrase:
    #    if token.is_punct or token.is_stop or token.text.lower() in spanish_stopwords:
    #        continue
    #    yield token.lemma_.lower()
    lemmas = [tok.lemma_ for tok in parsed_phrase if len(tok) > 3]

    return lemmas
    





filtered_sent=[]

#  "nlp" Object is used to create documents with linguistic annotations.

# filtering stop words
for word in doc:
    if word.is_stop==False:
        filtered_sent.append(word)
print("Filtered Sentence:",filtered_sent)


