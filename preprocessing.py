import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing import sequence, text


# read the dataframe
df = pd.read_csv('quora_duplicate_questions.tsv', sep='\t')


#drop unnecessary columns
df.drop(['id','qid1','qid2'], axis=1, inplace=True)


#define our preprocessing function
def clean_text(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"n't", "not", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\/", " / ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r"\0s", " 0 ", text)
    text = re.sub(r" 9 11 ", " 911 ", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r" j k ", " jk ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


#preprocess our questions
sents = []
for i in tqdm(range(len(df.question1))):
    sents.append(clean_text(df.question1[i]))
df['question1']  = sents

sents = []
for i in tqdm(range(len(df.question2))):
    sents.append(clean_text(df.question2[i]))
df['question2']  = sents

#pickle it to the root folder
with open ('question_pair.pickle', 'wb') as f:
    pickle.dump(df, f)

#
tk = text.Tokenizer(num_words=200000)
# we use keras Tokenizer to tokenizer the data. 
# we will only consider top 200000 words that occur in the dataset

tk.fit_on_texts(list(df.question1.values.astype(str)) + list(df.question2.values.astype(str)))
word_index = tk.word_index

#import our GloVe Embedding file
embeddings_index = {}
f = open('glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


#make our embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    

    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector



# sav3 the weight matrix for reuse
np.savetxt('embeddings.txt',embedding_matrix)
