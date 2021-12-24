from flask import Flask, request
import pickle
from senseed_detector import predict_ed
from keras.models import load_model

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data_obj = request.get_json()
    tweets_arr = data_obj['data']
    tweets = []

    for tweet_obj in tweets_arr:
        msg = tweet_obj['text_orig']
        tweets += [msg]

    with open("tokenizer.pk", 'rb') as tokenizer_file:
        tokenizer_obj = pickle.load(tokenizer_file)
    model = load_model("glove_model.h5")
    ed = 0

    for text in tweets:
        pred, score = predict_ed(text, tokenizer_obj, model)
        if pred == 1:
            ed += 1

    score = (ed / len(tweets)) * 100

    return str(score)


if __name__ == '__main__':
    app.run()
