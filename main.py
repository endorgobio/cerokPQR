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
import spacy
from spacy.lang.es import Spanish
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


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

# Remove all rows with NA in column Contenido
dfPQR = dfPQR.dropna(subset=['Contenido'])

# Filter NA in in column 'Riesgo de vida'  
dfPQR = dfPQR.dropna(subset=['Riesgo_vida'])


# Convert contenido to lowercase
dfPQR["Contenido"] = dfPQR["Contenido"].str.lower()

# Search for 'riesgo de vida' in the content
#dfPQR['riesgo_vida_expl'] = dfPQR['Contenido'].str.contains('riesgo de vida')


# Correct those that contains 'riesgo de vida' but were not labaled so
def explicit_riesgo(content, actual_riesgo):
    if actual_riesgo == 'SI':
        return actual_riesgo
    elif 'riesgo de vida' in str(content):#.contains('riesgo de vida'):
        return 'SI'
    else:
        return 'NO'
dfPQR['Riesgo_vida'] = dfPQR.apply(lambda x: explicit_riesgo(
    x['Contenido'], x['Riesgo_vida']), axis=1)

# Applying encoding to the Riesgo_vida column
#dfPQR['Riesgo_vida_encode'] = dfPQR['Riesgo_vida'].factorize()[0] 
dfPQR['Riesgo_vida_encode'] = [1 if x =='SI' else 0 for x in dfPQR['Riesgo_vida']]

#stop_words = list(stopwords.words('spanish'))
# Se añade la stopword: amp, ax, ex
#stop_words.extend(("amp", "xa", "xe"))

# Consider contenido as unique regresor
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
                        min_df     = 5,
                        stop_words = spanish_stopwords
                    )
#tfidf_vectorizador.fit(X_train)

#print(tfidf_vectorizador.get_feature_names_out())

weights = {0:1, 1:2.5}
classifier = LogisticRegression(class_weight=weights)

# Create pipeline using Bag of Words
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


confusion_matrix(y_true = y_test, y_pred= predicted)


frame = {'Contenido': X_test, 'y_real': y_test, 'y_hat': predicted }
  
result = pd.DataFrame(frame)

result['Riesgo_vida_textual'] = [1 if 'riesgo de vida' in str(x) else 0 for x in result['Contenido']]
result_grouped = result.groupby(['Riesgo_vida_textual', 'y_real', 'y_hat']).y_hat.agg(
    suma=('count')
)
  
print(result)



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
