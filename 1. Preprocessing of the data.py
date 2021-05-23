# import packages
import docx
import pandas as pd
import os
import numpy as np
import re
import string
import spacy

# LOAD DATA
os.chdir('/data')
path = '/data'
files = []
pseudo_name = []
x=1
for file in os.listdir(path):
    if file.endswith('.docx'):
        files.append(file)
        pseudo_name.append("company{x}".format(x=x))
        x=x+1

# order the Word files into lists of respectively smaller paragraphs and the whole documents
paragraphs = []
documents = []
for i in range(len(files)):
    doc = docx.Document(files[i])
    paragraphs.append([p.text for p in doc.paragraphs])
    documents.append(' '.join(p.text for p in doc.paragraphs))


# making the dataframe
# the df is used to create df2 where the text data is divided into paragrahs and related to the corresponding company
df = pd.DataFrame({'company':pseudo_name,'documents':documents,'paragraphs':paragraphs})
df2 = df.explode('paragraphs')

# remove empty rows
df2.replace("", float("NaN"), inplace=True)
df2.dropna(subset = ["paragraphs"], inplace=True)
# update the list of paragraphs
paragraphs = list(df2['paragraphs'])


###### CLEAN DATA
# load stop words file
f = open("stopord.txt", "r")
stop_list = list(f.read().split('\n'))
f.close()

stop_list.extend([words.lower() for words in list(''.split(", "))]) # here you can add your own items to the stop list
not_stop = [] # here you can add items to remove from the stop list
stop_list = [word for word in stop_list if not word in not_stop] # the final stop list 

# function to remove stopwords
def clear_stopwords(text, stop_words):
    textword = text.split()
    resultwords  = [word for word in textword if word not in stop_words]
    return ' '.join(resultwords)

# function to lemmatize
nlp = spacy.load("da_core_news_lg")
def lemmatize(sentence):
    doc = nlp(sentence)
    lemmas = [token.lemma_ for token in doc]
    all_lemmas = [item for item in lemmas]
    return all_lemmas

#lowercase, remove punctuations, stopwords, and lemmatize the paragraphs
lower_text = []
for sent in paragraphs:
    sent = sent.replace('–', ' ')
    sent = sent.replace('”', ' ')
    lower_text.append(' '.join([words.lower() for words in sent.split()]))

text_no_punc = [sent.translate(str.maketrans('', '', string.punctuation)) for sent in lower_text]
text_no_num = [sent.translate(str.maketrans('', '', string.digits)) for sent in text_no_punc]
cleaned1 = [clear_stopwords(sent,stop_list) for sent in text_no_num]
lemmatized = [' '.join(lemmatize(i)) for i in cleaned1]
cleaned = [clear_stopwords(sent,stop_list) for sent in lemmatized]

# add the cleaned and lemmatized data to the dataframe
df2['cleaned'] = cleaned
df2['lemma'] = lemmatized

###### SAVE DATA
df2.to_csv('/data.csv') # add directory of where to save the file
