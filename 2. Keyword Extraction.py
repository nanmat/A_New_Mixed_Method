import pandas as pd
import os
import numpy as np
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from danlp.models import load_spacy_model
from collections import Counter
from nltk.tokenize import word_tokenize
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# load data
os.chdir('/data')
df = pd.read_csv('/data.csv')

# set path to where you want to save the plots
os.chdir('/plots')
# WORDCLOUDS

# function to make wordcloud 
def make_wordcloud(topics,company):
    wordcloud = WordCloud(background_color='white',colormap='gnuplot',width=800, height=500,random_state=21, max_font_size=110).generate_from_frequencies(topics)
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(str(company),weight='bold', fontsize=20)
    plt.savefig("wordcloud_{x}.png".format(x=company), dpi=600)    
        
#### WORD CLOUDS FOR EACH COMPANY
for i,j in zip(df['cleaned'],df['company']):
    all_words = ''.join([word for word in i])
    tokens = all_words.split(" ")
    word_counts = Counter(tokens)
    make_wordcloud(word_counts, j) 


# function to get n_grams from a corpus
def get_top_ngrams(corpus,n_gram=1, n=10):
  vec = CountVectorizer(ngram_range=(n_gram, n_gram), stop_words=None).fit(corpus)
  bag_of_words = vec.transform(corpus)
  sum_words = bag_of_words.sum(axis=0) 
  words_freq = [(word, sum_words[0, idx]) for word, idx in   vec.vocabulary_.items()]
  words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
  return words_freq[:n]


### TOP BIGRAMS
bigrams = get_top_ngrams(df['cleaned'], 2, 15)
df_bigram = pd.DataFrame(bigrams, columns = ['sentence' , 'count'])

# plot the bigrams in a barplot 
bigram_plot = go.Figure()
bigram_plot.add_trace(go.Bar(x=df_bigram['sentence'],
                               y=df_bigram['count'],
                               orientation='v',
                               texttemplate="%{y}",
                               textposition="inside",
                               hovertemplate="%{y}: %{x}"))

bigram_plot.update_traces(opacity=0.8)
bigram_plot.update_yaxes(title_text='Occurrences') 
bigram_plot.update_layout(barmode='relative',
                        height=600,
                        width=900,
                        bargap=0.3,
                        colorway=['#7061ce'],
                        legend_orientation='v',
                        legend_x=1, legend_y=0,
                        title_text= 'Most frequent bigrams')
bigram_plot.show()
bigram_plot.write_html("/interactive_bigram.html") # save the interactive plot as a html file

# TOP KEYWORDS DIVIDED INTO POS-TAGS
# load model to pos-tag and function to extract the pos-tags to each word
nlp = load_spacy_model()

def prepare_the_text(df_tekst):
    doc = nlp(' '.join(df_tekst.map(lambda x: str(x))))
    return doc

def get_pos_word(doc, pos1='NOUN', pos2='ADJ',pos3='VERB'): # you can choose other pos-tags here
    words_pos1=[token.text for token in doc if token.pos_==pos1]
    words_pos2=[token.text  for token in doc if token.pos_==pos2]
    words_pos3=[token.text  for token in doc if token.pos_==pos3] 
    return words_pos1,words_pos2, words_pos3

# use the functions
doc = prepare_the_text(df.cleaned)
words1, words2, words3 = get_pos_word(prepare_the_text(df.cleaned))
# get the most frequent words in each category by use og the function to get n_grams. 
nouns = get_top_ngrams(words1,n_gram=1, n=10)
adjectives = get_top_ngrams(words2,n_gram=1, n=10)
verbs = get_top_ngrams(words3,n_gram=1, n=10)
nouns = pd.DataFrame(nouns, columns = ['word' , 'count'])
adjectives = pd.DataFrame(adjectives, columns = ['word' , 'count'])
verbs = pd.DataFrame(verbs, columns = ['word' , 'count'])

# plot the top keywords of the three pos-tags beside each other
frequent_words_plot = make_subplots(rows=1, cols=3,
                        subplot_titles=("Nouns", "Verbs", "Adjectives"),
                        shared_yaxes=True)

frequent_words_plot.add_trace(go.Bar(x=nouns['word'],
            y=nouns['count'],
            orientation='v',
            texttemplate="%{y}",
            textposition="inside",
            hovertemplate="%{y}: %{x}"),
            row=1, col=1)

frequent_words_plot.add_trace(go.Bar(x=verbs['word'],
            y=verbs['count'],
            orientation='v',
            texttemplate="%{y}",
            textposition="inside",
            hovertemplate="%{y}: %{x}"),
            row=1, col=2)

frequent_words_plot.add_trace(go.Bar(x=adjectives['word'],
            y=adjectives['count'],
            orientation='v',
            texttemplate="%{y}",
            textposition="inside",
            hovertemplate="%{y}: %{x}"),
            row=1, col=3)

frequent_words_plot.update_traces(opacity=0.8)
frequent_words_plot.update_yaxes(title_text='Occurrences') 

frequent_words_plot.update_layout(height=600, width=1300, 
                    title_text="Most frequent nouns, verbs and adjectives",
                    colorway=['#8f160d', '#cf654b', '#ffb495'],
                    showlegend=False)
frequent_words_plot.show()
frequent_words_plot.write_html("/interactive_frequentwordsplot.html") # save the interactive plot as a html file
