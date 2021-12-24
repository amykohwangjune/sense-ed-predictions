import numpy as np
import pandas as pd
import re
import string
import pickle
import nltk
import tensorflow
import keras
import gingerit
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from gingerit.gingerit import GingerIt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# Importing the dataset
def read_dataset(filename):
    tweets = pd.read_csv(filename)
    tweets.head()
    tweets.info()
    return tweets


def spell_check(text):
    return GingerIt().parse(text)['result']


def clean_text(text):
    text = text.lower()

    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = pattern.sub('', text)
    text = " ".join(filter(lambda x: x[0] != '@', text.split()))
    emoji = re.compile("["
                       u"\U0001F600-\U0001FFFF"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       "]+", flags=re.UNICODE)

    text = emoji.sub(r'', text)
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)
    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    return text


def CleanTokenize(df):
    head_lines = list()
    lines = df["text_orig"].values.tolist()
    for line in lines:
        line = clean_text(line)
        # tokenize the text
        tokens = word_tokenize(line)
        # remove puntuations
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove non alphabetic characters
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not")
        # remove stop words
        words = [w for w in words if not w in stop_words]
        head_lines.append(words)
    return head_lines


def pretrained_model(data, head_lines):
    validation_split = 0.2
    max_length = 25

    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(head_lines)
    sequences = tokenizer_obj.texts_to_sequences(head_lines)

    word_index = tokenizer_obj.word_index
    print("unique tokens - ", len(word_index))
    vocab_size = len(tokenizer_obj.word_index) + 1
    print('vocab size -', vocab_size)

    lines_pad = pad_sequences(sequences, maxlen=max_length, padding='post')
    ed = data['EDPatience'].values

    indices = np.arange(lines_pad.shape[0])
    np.random.shuffle(indices)
    lines_pad = lines_pad[indices]
    ed = ed[indices]

    num_validation_samples = int(validation_split * lines_pad.shape[0])

    X_train_pad = lines_pad[:-num_validation_samples]
    y_train = ed[:-num_validation_samples]
    X_test_pad = lines_pad[-num_validation_samples:]
    y_test = ed[-num_validation_samples:]

    print('Shape of X_train_pad:', X_train_pad.shape)
    print('Shape of y_train:', y_train.shape)

    print('Shape of X_test_pad:', X_test_pad.shape)
    print('Shape of y_test:', y_test.shape)

    embeddings_index = {}
    embedding_dim = 100
    f = open('glove.twitter.27B.100d.txt', encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    c = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            c += 1
            embedding_matrix[i] = embedding_vector
    print(c)

    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_length,
                                trainable=False)

    return embedding_layer, tokenizer_obj, X_train_pad, y_train, X_test_pad, y_test


def train_glove_model(embedding_layer, X_train_pad, y_train, X_test_pad, y_test):
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(5, dropout=0.5, recurrent_dropout=0.35))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_pad, y_train, batch_size=32, epochs=10, validation_data=(X_test_pad, y_test), verbose=0)
    loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
    print(acc * 100)
    return model


def train_models():
    data = read_dataset('ed-data_augmented.csv')
    head_lines = CleanTokenize(data)
    glove_embedding_layer, tokenizer_obj, X_train_pad, y_train, X_test_pad, y_test = pretrained_model(data, head_lines)
    model = train_glove_model(glove_embedding_layer, X_train_pad, y_train, X_test_pad, y_test)
    return model, tokenizer_obj


def save_models(model, tokenizer_obj):
    with open('tokenizer.pk', 'wb') as tvec:
        pickle.dump(tokenizer_obj, tvec)

    model.save("glove_model.h5")


def predict_ed(s, tokenizer_obj, model):
    max_length = 25
    # if detect_en(s) != True:
    #     return 'Not an English sentence/ word!'
    if s == "":
        return 0
    x_final = spell_check(s)
    x_final = pd.DataFrame({"text_orig": [x_final]})
    test_lines = CleanTokenize(x_final)
    test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    test_review_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')
    pred = model.predict(test_review_pad)
    pred *= 100
    if pred[0][0] >= 50:
        return 1, "{:.2f}".format(pred[0][0])
    else:
        return 0, "{:.2f}".format(pred[0][0])

# model, tokenizer_obj = train_models()
# save_models(model, tokenizer_obj)
