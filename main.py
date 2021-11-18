# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 18:22:39 2021

@author: pmayaduque
"""

import pandas as pd
import wget
import numpy as np
import re

# Preprocesado y modelado
# ==============================================================================
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

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
#dfPQR_filtered = dfPQR_filtered[dfPQR_filtered['Riesgo de vida'].notna()]

# Convert contenido to lowercase
dfPQR["Contenido"] = dfPQR["Contenido"].str.lower()

# Search for 'riesgo de vida' in the content
dfPQR['riesgo_vida_expl'] = dfPQR['Contenido'].str.contains('riesgo de vida')
dfPQR_riesgo = dfPQR[dfPQR['Riesgo_vida']=='SI']
dfPQR_riesgo['riesgo_vida_expl'].sum()

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
dfPQR['Riesgo_vida_encode'] = dfPQR['Riesgo_vida'].factorize()[0] 

stop_words = list(stopwords.words('spanish'))
# Se añade la stopword: amp, ax, ex
stop_words.extend(("amp", "xa", "xe"))

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
    # Tokenización por palabras individuales
    nuevo_texto = nuevo_texto.split(sep = ' ')
    # Eliminación de tokens con una longitud < 2
    nuevo_texto = [token for token in nuevo_texto if len(token) > 1]
    
    return(nuevo_texto)



tfidf_vectorizador = TfidfVectorizer(    
                        tokenizer  = limpiar_tokenizar,
                        min_df     = 3,
                        stop_words = stop_words
                    )
tfidf_vectorizador.fit(X_train)

print(tfidf_vectorizador.get_feature_names())







