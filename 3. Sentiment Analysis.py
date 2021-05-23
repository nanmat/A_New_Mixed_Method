# load libaries
import pandas as pd
import os
import numpy as np
from nltk.tokenize import sent_tokenize
import random
from danlp.models import load_bert_emotion_model, load_bert_tone_model,load_spacy_model
import operator 
import plotly.graph_objects as go
import plotly.express as px

# load data
os.chdir('/data')
df = pd.read_csv('data.csv')

# divide documents into sentences
s = df["paragraphs"].apply(lambda x : sent_tokenize(x)).apply(pd.Series,1).stack()
s.index = s.index.droplevel(-1) # match the index of df
s.name = 'text_sentence' # give name to join

# removing empty rows
s.replace('', np.nan, inplace=True)
s.dropna(inplace=True)

# new dataframe
del df["paragraphs"]
df = df.join(s)
df.head(10)

### ESTIMATE EMOTIONS, POLARITY AND TONE
# load classiers and build function with them
classifier_emo = load_bert_emotion_model()
classifier_tone = load_bert_tone_model()

def predict_emo(x):
    return classifier_emo.predict(x)

def predict_tone(x):
    return classifier_tone.predict(x)

# BERT EMOTION
emotion = [predict_emo(i) for i in df['text_sentence']]
# get the percentage prediction for each emotion
Glede_Sindsro = []
Tillid_Accept= []
Forventning_Interrese= []
Overasket_Målløs= []
Vrede_Irritation= []
Foragt_Modvilje= []
Sorg_trist= []
Frygt_Bekymret= []
for i in df['text_sentence']:
    result = classifier_emo.predict_proba(i)
    Glede_Sindsro.append(result[0][0])
    Tillid_Accept.append(result[0][1])
    Forventning_Interrese.append(result[0][2])
    Overasket_Målløs.append(result[0][3])
    Vrede_Irritation.append(result[0][4])
    Foragt_Modvilje.append(result[0][5])
    Sorg_trist.append(result[0][6])
    Frygt_Bekymret.append(result[0][7])


# BERT TONE:    
d = []
for i in df['text_sentence']:
    d.append(predict_tone(i))
tone = [i['analytic'] for i in d]
polarity = [i['polarity'] for i in d]

# get the percentage prediction for each tone and polarity
positiv = []
negativ = []
neutral = []
subjektiv = []
objektiv = []
for i in df['text_sentence']:
    result = classifier_tone.predict_proba(i)
    positiv.append(result[0][0])
    neutral.append(result[0][1])
    negativ.append(result[0][2])
    objektiv.append(result[1][0])
    subjektiv.append(result[1][1])


# Add all classifications to dataframe
df['polarity']              = polarity
df['tone']                  = tone
df['emotion']               = emotion

df['Glede_Sindsro']         = Glede_Sindsro 
df['Tillid_Accept']         = Tillid_Accept
df['Forventning_Interesse'] = Forventning_Interrese
df['Overrasket_Målløs']     = Overasket_Målløs
df['Vrede_Irritation']      = Vrede_Irritation
df['Foragt_Modvilje']       = Foragt_Modvilje
df['Sorg_Trist']            = Sorg_trist
df['Frygt_Bekymret']        = Frygt_Bekymret

df['positiv'] = positiv
df['neutral'] =neutral
df['negativ'] =negativ
df['subjektiv'] =subjektiv
df['objektiv'] =objektiv

# save dataframe
df.to_csv('data_sentiment.csv')

### SHOW SENTENCES WITH HIGHEST SCORES FOR EACH CLASS
pd.options.display.max_colwidth = 1000

# POLARITY
for index,review in enumerate(df.iloc[df['positive'].sort_values(ascending=False)[:5].index]['text_sentence']):
  print('Eksempel {}:\n'.format(index+1),review)

for index,review in enumerate(df.iloc[df['negative'].sort_values(ascending=False)[:5].index]['text_sentence']):
  print('Eksempel {}:\n'.format(index+1),review)

for index,review in enumerate(df.iloc[df['neutral'].sort_values(ascending=False)[:5].index]['text_sentence']):
  print('Eksempel {}:\n'.format(index+1),review)

# TONE
for index,review in enumerate(df.iloc[df['subjective'].sort_values(ascending=False)[:5].index]['text_sentence']):
  print('Eksempel {}:\n'.format(index+1),review)

for index,review in enumerate(df.iloc[df['objective'].sort_values(ascending=False)[:5].index]['text_sentence']):
  print('Eksempel {}:\n'.format(index+1),review)

# EMOTION
for index,review in enumerate(df.iloc[df['Vrede_Irritation'].sort_values(ascending=False)[:5].index]['text_sentence']):
  print('Eksempel {}:\n'.format(index+1),review)

for index,review in enumerate(df.iloc[df['Tillid_Accept'].sort_values(ascending=False)[:5].index]['text_sentence']):
  print('Eksempel {}:\n'.format(index+1),review)

for index,review in enumerate(df.iloc[df['Forventning_Interesse'].sort_values(ascending=False)[:5].index]['text_sentence']):
  print('Eksempel {}:\n'.format(index+1),review)

for index,review in enumerate(df.iloc[df['Foragt_Modvilje'].sort_values(ascending=False)[:5].index]['text_sentence']):
  print('Eksempel {}:\n'.format(index+1),review)

for index,review in enumerate(df.iloc[df['Glede_Sindsro'].sort_values(ascending=False)[:5].index]['text_sentence']):
  print('Eksempel {}:\n'.format(index+1),review)

for index,review in enumerate(df.iloc[df['Sorg_Trist'].sort_values(ascending=False)[:5].index]['text_sentence']):
  print('Eksempel {}:\n'.format(index+1),review)

for index,review in enumerate(df.iloc[df['Overrasket_Målløs'].sort_values(ascending=False)[:5].index]['text_sentence']):
  print('Eksempel {}:\n'.format(index+1),review)

for index,review in enumerate(df.iloc[df['Frygt_Bekymret'].sort_values(ascending=False)[:5].index]['text_sentence']):
  print('Eksempel {}:\n'.format(index+1),review)
  
  
###### PLOTTING ########
os.chdir('/plots') # set directory to the folder where you want to save the plots
#change labels to english
df['emotion'] = df['emotion'].replace(['Forventning/Interrese','No Emotion','Foragt/Modvilje','Tillid/Accept','Frygt/Bekymret','Vrede/Irritation','Overasket/Målløs','Glæde/Sindsro','Sorg/trist'],
                    ['Expectation','No Emotion','Contempt','Trust','Fear','Anger','Surprise','Joy','Sadness'])

### PREPARE DATA FOR PLOTTING
new1 = df[['company','emotion']].copy()
new_emotion = pd.crosstab(new1['company'], new1['emotion'], normalize='index') * 100
new_emotion = new_emotion.reindex(df2.company, axis="rows")
new_emotion = new_emotion.round(1)

new2 = df[['company','polarity']].copy()
new_polarity = pd.crosstab(new2['company'], new2['polarity'], normalize='index') * 100
new_polarity = new_polarity.reindex(df2.company, axis="rows")
new_polarity = new_polarity.round(1)

new3 = df[['company','tone']].copy()
new_tone = pd.crosstab(new3['company'], new3['tone'], normalize='index') * 100
new_tone = new_tone.reindex(df2.company, axis="rows")
new_tone = new_tone.round(1)

###################### PLOTTING

# POLARITY
new_polarity.negative = new_polarity.negative * -1 # making negative, in order to plot in the other direction
polarity_plot = go.Figure()
  
# Iterating over the columns
for col in new_polarity.columns:  
    # Adding a trace for negative sentiment
    polarity_plot.add_trace(go.Bar(x=-new_polarity[col].values,
                               y=new_polarity.index,
                               orientation='h',
                               name=col,
                               customdata=new_polarity[col],
                               texttemplate="%{x} %",
                               textposition="inside",
                               hovertemplate="%{y}: %{customdata}"))

for col in new_polarity.columns: 
    # Adding a trace for positive and neutral sentiment
    polarity_plot.add_trace(go.Bar(x=new_polarity[col],
                               y=new_polarity.index,
                               orientation='h',
                               name=col,
                               texttemplate="%{x} %",
                               textposition="inside",
                               hovertemplate="%{y}: %{x}"))  

# change the richness of color of traces and x-axis
polarity_plot.update_traces(opacity=0.8)
polarity_plot.update_xaxes(title_text='%') 

# Specify the layout
polarity_plot.update_layout(barmode='relative',
                        height=1000,
                        width=1000,
                        yaxis_autorange='reversed',
                        bargap=0.3,
                        colorway=['#8f160d', '#cf654b', '#ffb495'],
                        plot_bgcolor='#ffffff',
                        legend_orientation='v',
                        legend_x=1, legend_y=0,
                        title_text= 'Percentage distribution of polarity'
                        )
# show and save the plot
polarity_plot.show()
polarity_plot.write_html("interactive_polarity.html")


# TONE
tone_plot = go.Figure()

for col in new_tone.columns: 
    # Adding a trace for subjective and objective sentiment
    tone_plot.add_trace(go.Bar(x=new_tone[col],
                               y=new_tone.index,
                               orientation='h',
                               name=col,
                               texttemplate="%{x} %",
                               textposition="inside",
                               hovertemplate="%{y}: %{x}"))

# change the richness of color of traces and x-axis
tone_plot.update_traces(opacity=0.8)
tone_plot.update_xaxes(title_text='%') 

# Specify the layout
tone_plot.update_layout(barmode='relative',
                        height=1000,
                        width=900,
                        yaxis_autorange='reversed',
                        bargap=0.3,
                        colorway=['#4a3fad', '#c9a9f1'],
                        plot_bgcolor='#ffffff',
                        legend_orientation='v',
                        legend_x=1, legend_y=0,
                        title_text= 'Percentage distribution of tone'
                        )
# show and save the plot
tone_plot.show()
tone_plot.write_html("interactive_tone.html")



# EMOTION
column_name = ['Expectation','No emotion','Contempt','Trust','Fear','Anger','Surprise','Joy','Sadness'] # order the emotions
new_emotion = new_emotion[column_name] 
emotion_plot = go.Figure()

for col in new_emotion.columns: 
    # Adding a trace for all emotions
    emotion_plot.add_trace(go.Bar(x=new_emotion[col],
                               y=new_emotion.index,
                               orientation='h',
                               name=col,
                               texttemplate="%{x} %",
                               textposition="inside",
                               hovertemplate="%{y}: %{x}"))

# change the richness of color of traces and x-axis
emotion_plot.update_traces(opacity=0.8)
emotion_plot.update_xaxes(title_text='%') 

# Specify the layout
emotion_plot.update_layout(barmode='relative',
                        height=1000,
                        width=1400,
                        yaxis_autorange='reversed',
                        bargap=0.3,
                        colorway=['#4a3fad', '#7061ce', '#9b84e8', '#c9a9f1', '#ffcfe4', '#ffbcaf', '#f4777f', '#cf3759', '#93003a'],
                        plot_bgcolor='#ffffff',
                        legend_orientation='v',
                        legend_x=1, legend_y=0,
                        title_text= 'Percentage distribution of emotions'
                        )
# show and save the plot
emotion_plot.show()
emotion_plot.write_html("interactive_emotion.html")

# EMOTION OVERALL - CIRCLE PLOT
emotion_overall_plot = px.pie(df, names='emotion', title='Overall percentage distribution of emotions',color='emotion',
                color_discrete_map={'Expectation':'#4a3fad',
                                    'No emotion':'#7061ce',
                                    'Contempt':'#9b84e8',
                                    'Trust':'#c9a9f1',
                                    'Fear':'#ffcfe4',
                                    'Anger':'#ffbcaf',
                                    'Surprise':'#f4777f',
                                    'Joy':'#cf3759',
                                    'Sadness':'#93003a'},
                                    height=600,
                                    width=600)
emotion_overall_plot.update_traces(textposition='inside', textinfo='percent+label')
emotion_overall_plot.update_layout(showlegend=False)

emotion_overall_plot.show()
emotion_overall_plot.write_html("interactive_emotion_overall.html")
  
  
