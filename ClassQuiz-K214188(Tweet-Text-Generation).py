#!/usr/bin/env python
# coding: utf-8

# In[383]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import re
import chardet    

#import re
#import nltk
#import string
#from nltk.stem import WordNetLemmatizer
#from nltk.corpus import stopwords
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')
#nltk.download('wordnet')
#nltk.download('stopwords')


# In[384]:


#df = pd.read_csv("realDonaldTrump_in_office.csv")
#df = pd.read_csv("realDonaldTrump_in_office.csv")
with open("realDonaldTrump_in_office.csv", 'r', encoding='utf8') as temp_f:
    # get No of columns in each line
    col_count = [ len(l.split(",")) for l in temp_f.readlines() ]

### Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1)
column_names = [i for i in range(0, max(col_count))]

### Read csv
df = pd.read_csv("realDonaldTrump_in_office.csv", header=None, delimiter=",", names=column_names)


# In[385]:


df.head()


# In[386]:


df['final_tweet'] = df[df.columns[3:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)


# In[387]:


df['final_tweet']


# In[388]:


#Remove URLs


def cleaning_PicURL (text):
    text = re.sub(r'pic.twitter.com/[\w]*',"", text)
    return text

#f['text'] = df['text'].apply(lambda x: cleaning_PicURL(x))



temp_list = []
for index, row in df.iterrows():
    #print(row['final_tweet'])
    text = cleaning_PicURL(row['final_tweet'])
    text = re.sub(r'https?:\/\/\S*', '', text, flags=re.MULTILINE)
    temp_list.append(text)
    
temp_list
df['removeURL'] = temp_list
df['removeURL']



#text = re.sub(r'https?:\/\/\S*', '', df['final_tweet'][11], flags=re.MULTILINE)
#text 
df['removeURL'] = df['removeURL'].replace(r'[^0-9a-zA-Z ]', '', regex=True).replace("'", '')
df.dropna(subset=['removeURL'], inplace=True)
df


# In[389]:


text = df['removeURL'].str.lower()

print('max tweet len:',text.map(len).max())
print('min tweet len:',text.map(len).min())
text.map(len).hist();


# In[390]:


chars = sorted(list(set(''.join(text))))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# In[391]:


chars


# In[392]:


for c in chars[-19:]:
    print('\nCHAR:', c)
    smple = [x for x in text if c in x]
    print(random.sample(smple,min(3,len(smple))))


# In[393]:


# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 1
sentences = []
next_chars = []
for x in text:
    for i in range(0, len(x) - maxlen, step):
        sentences.append(x[i: i + maxlen])
        next_chars.append(x[i + maxlen])
print('nb sequences:', len(sentences))


# In[394]:


## check example
for i in range(3):
    print(sentences[i],'==>',next_chars[i])


# In[395]:


text[1]


# In[396]:


print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# # build the model: a single LSTM¶
# 

# In[397]:


print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

# optimizer = RMSprop(lr=0.01)
optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# #  Sampler

# In[398]:


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[399]:


for temperature in [0.1, 0.2, 0.3,  0.5, 1.0, 1.2, 1.3]:
    print(sample([.1,.3,.5,.1],temperature=temperature))


# # Generate Text at Epoch End

# In[400]:


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)
    
#     start_index = random.randint(0, len(text) - maxlen - 1)
    tweet = np.random.choice(text) # select random tweet
    start_index = 0

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = tweet[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(120):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


# In[ ]:


epochs = 1

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=epochs,
          callbacks=[print_callback])


# In[ ]:


print('Build model...')
model2 = Sequential()
model2.add(LSTM(128, input_shape=(maxlen, len(chars)),return_sequences=True))
model2.add(Dropout(0.2))
model2.add(LSTM(128))
model2.add(Dropout(0.2))
model2.add(Dense(len(chars), activation='softmax'))

# optimizer = RMSprop(lr=0.01)
optimizer = Adam()
model2.compile(loss='categorical_crossentropy', optimizer=optimizer)


# # Print Test Sentence¶
# 

# In[ ]:


def generate_w_seed(sentence,diversity):
    sentence = sentence[0:maxlen]
    print(f'seed: {sentence}')
    print(f'diversity: {diversity}')
    generated = ''
    generated += sentence
    
    sys.stdout.write(generated)

    for i in range(120):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
    return


# In[ ]:


for s in random.sample(list(text),5):
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        generate_w_seed(s,diversity)
        print()

