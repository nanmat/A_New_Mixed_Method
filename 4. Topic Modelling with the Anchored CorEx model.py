import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct 
import os
import pandas as pd
from nltk.tokenize import word_tokenize 
from corextopic import vis_topic as vt
from sklearn.feature_extraction.text import CountVectorizer
import _pickle as cPickle

# LOAD DATA 
os.chdir('/data')
df = pd.read_csv('data.csv')
df.dropna(subset = ["cleaned_paragraphs"], inplace=True)

# transform data
vectorizer = CountVectorizer(analyzer='word',binary=True) 
doc_word = vectorizer.fit_transform(df['cleaned_paragraphs'])
doc_word = ss.csr_matrix(doc_word)

words = list(np.asarray(vectorizer.get_feature_names()))
company = list(df['company'])
docs = list(df['paragraphs'])

doc_word.shape # n_docs x n_words

# FUNCTIONS TO ACCESS THE TERMS AND DOCUMENTS FROM THE COREX MODELS
def corex_topics(topic_model):
    topics = topic_model.get_topics()
    for n,topic in enumerate(topics):
        topic_words,_ = zip(*topic)
        print('Topic {} keywords: '.format(n) + ', '.join(topic_words))

def show_top_doc(topic_model, num_to_show):
    topics = topic_model.get_topics()
    for i in range(len(topics)):
        print(f'Topic {i}')
        top_doc = [doc[0] for doc in topic_model.get_top_docs(topic=i, n_docs=num_to_show)]
        for i in range(len(top_doc)):
            print(f"{i+1}) {top_doc[i]}")
        print("")

#############################################
############# THE COREX MODEL ###############
#############################################


# Train the CorEx topic model on 70 topics to figure out how many topics to have
topic_model = ct.Corex(n_hidden=70, seed=2, max_iter=200)
topic_model.fit(doc_word, words=words, docs=docs)

# set directory for saving plots and plot the pointwise total correlation for 70 topics    
os.chdir('/plots')
plt.figure()
plt.bar(range(tcs[x].shape[0]), tcs[x], color='#4e79a7', width=0.5)
plt.xlabel('Topic', fontsize=16)
plt.ylabel('Total Correlation (nats)', fontsize=16)
plt.savefig("topic_selection.png", dpi=300)


# after inspection, 32 topics are settled on. 
# now we optimize by testing different values for seed.
n_seed = [1,2,3,4,5,6,7,8,9,10]
tc = []
for i in n_seed:
    topic_model = ct.Corex(n_hidden=32, seed=i)
    topic_model.fit(doc_word, words=words, docs=docs)
    tc.append(topic_model.tc)

# print the total correlation score for each model
for i,j in zip(n_seed, tc):    
    print('\nTotal Correlation Score: ', j, 'for', i, 'seed number')


###### FINAL COREX MODEL: 32 topics and seed=4 (12.840217410847965 tc)
# run the model
corex_model = ct.Corex(n_hidden=32, seed=4)  # Define the number of latent (hidden) topics to use
corex_model.fit(doc_word, words=words, docs=docs) # gives the company

# plot the total correlation for the model with 32 topics
os.chdir('/plots')
plot = plt.bar(range(corex_model.tcs.shape[0]), corex_model.tcs, color='#4e79a7', width=0.5)
plt.xlabel('Topic', fontsize=16)
plt.ylabel('Total Correlation (nats)', fontsize=16)
plt.savefig('Selection_of_topics_32.png',dpi=600)

# Print the terms making up the topics 
corex_topics(corex_model)

# print the top paragraphs belonging to each topic
show_top_doc(corex_model, 3)


### SAVE THE MODEL
os.chdir('/models')
cPickle.dump(corex_model, open('corex_model_32.pkl', 'wb'))
### LOAD OLD MODEL
topic_model = cPickle.load(open('corex_model_32.pkl', 'rb'))


#############################################
############# ANCHOR COREX MODELS ###########
#############################################

# Anchor the lists of words 
anchor_frequentkeywords = [[],[],[]] # add most frequent keywords. each list inside correspond to one topic

anchor_domainknowledge = [[,],[,],[,]] # add domain knowledge keywords. 

# anchor strength above 1, 1.5-3 nudge the words into the topic, above 5 stronly enforces to find topics with the words
# keeping the same number of topics as in the CorEx - 32
# TEST FOR OPTIMAL SEED VALUE
n_seed = [1,2,3,4,5,6,7,8,9,10]
tc = []
for i in n_seed:
    topic_model = ct.Corex(n_hidden=32, seed=i)  # Define the number of latent (hidden) topics to use.
    topic_model.fit(doc_word, words=words, docs = docs, anchors=anchor_frequentkeywords, anchor_strength=4)
    tc.append(topic_model.tc)

for i,j in zip(n_seed, tc):    
    print('\nTotal Correlation Score: ', j, 'for', i, 'seed number')
# BEST RESULTS: domain knowledge: 50.26645935341452 for 9 seed number

tc = []
for i in n_seed:
    topic_model = ct.Corex(n_hidden=32, seed=i)  # Define the number of latent (hidden) topics to use.
    topic_model.fit(doc_word, words=words, docs = docs, anchors=anchor_domainknowledge, anchor_strength=4)
    tc.append(topic_model.tc)

for i,j in zip(n_seed, tc):    
    print('\nTotal Correlation Score: ', j, 'for', i, 'seed number')
# BEST RESULTS: 27.733828765214422 for 2 seed number

# RUN THE FINAL MODELS
anchored_frequentkeywords_model = ct.Corex(n_hidden=32, seed=9)
anchored_frequentkeywords_model.fit(doc_word, words=words, docs = docs, anchors=anchor_frequentkeywords, anchor_strength=4)

anchored_domainknowledge_model = ct.Corex(n_hidden=32, seed=2)
anchored_domainknowledge_model.fit(doc_word, words=words, docs = docs, anchors=anchor_domain_knowledge, anchor_strength=4)

# PLOT the pointwise total correlation
plt.figure()
plot = plt.bar(range(anchored_frequentkeywords_model.tcs.shape[0]), anchored_frequentkeywords_model.tcs, color='#4e79a7', width=0.5)
plt.xlabel('Topic', fontsize=16)
plt.ylabel('Total Correlation (nats)', fontsize=16)
plt.savefig('Selection_keywords.png',dpi=600)

plt.figure()
plot = plt.bar(range(anchored_domainknowledge_model.tcs.shape[0]), anchored_domainknowledge_model.tcs, color='#4e79a7', width=0.5)
plt.xlabel('Topic', fontsize=16)
plt.ylabel('Total Correlation (nats)', fontsize=16)
plt.savefig('Selection_themes.png',dpi=600)

# get the terms of each topic for each model
corex_topics(anchored_frequentkeywords_model)
corex_topics(anchored_domainknowledge_model)

# get the top documents belonging to each topic for each model
show_top_doc(anchored_frequentkeywords_model, 3)
show_top_doc(anchored_domainknowledge_model, 3)


### SAVE THE MODELs
os.chdir('/Users/admin/thesis_code/cs_seededTM_code/models')
cPickle.dump(anchored_frequentkeywords_model, open('anchored_keywords_model.pkl', 'wb'))
cPickle.dump(anchored_domainknowledge_model, open('anchored_domainknowledge_model.pkl', 'wb'))

### LOAD OLD MODELs
anchored_frequentkeywords_model = cPickle.load(open('anchored_frequentkeywords_model.pkl', 'rb'))
anchored_domainknowledge_model = cPickle.load(open('anchored_domainknowledge_model.pkl', 'rb'))
