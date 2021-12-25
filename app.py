from flask import Flask, request
import pickle
from keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data_obj = request.get_json()
    tweets_arr = data_obj['data']
    tweets = []

    for tweet_obj in tweets_arr:
        msg = tweet_obj['text']
        tweets += [msg]

    with open("tokenizer.pk", 'rb') as tokenizer_file:
        tokenizer_obj = pickle.load(tokenizer_file)
    model = load_model("glove_model.h5")
    ed = 0

    test_sequences = tokenizer_obj.texts_to_sequences(tweets)
    test_review_pad = pad_sequences(test_sequences, maxlen=25, padding='post')
    predictions = model.predict(test_review_pad)

    ed, score = 0, 0

    for pred in predictions:
        if pred >= 0.5:
            ed += 1

    score = 100 * ed / len(tweets)

    return str(score)



if __name__ == '__main__':
    app.run()
