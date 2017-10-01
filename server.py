import os
import numpy as np
from flask import Flask
from flask import request, render_template, session
from flask_script import Manager, Shell, Server
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import Required
from keras.models import load_model
from komma_dev.parsing import StringParser
from keras.preprocessing import sequence
#----------------------------------------------------------------------------------------
# Configuring the app
#----------------------------------------------------------------------------------------
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

bootstrap = Bootstrap(app)
manager = Manager(app)

def make_shell_context():
    return dict(app=app)

manager.add_command("shell", Shell(make_context=make_shell_context))
manager.add_command("local", Server(host="0.0.0.0", port=5000), Debug=True)
manager.add_command("runserver", Server(host="0.0.0.0", port=80))

#----------------------------------------------------------------------------------------
# Defining classes
#----------------------------------------------------------------------------------------
class NameForm(FlaskForm):
    text = TextAreaField('')
    submit = SubmitField('Submit')

#----------------------------------------------------------------------------------------
# Initializing models
#----------------------------------------------------------------------------------------
def initialize_models():
    print('Loading the vocabulary and model')
    vocab = np.load('models/vocab_50k.npy').item()
    model = load_model('models/model_50k_vocab.h5')
    threshold = 0.39
    MAX_CHUNKS = 50
    return vocab, model, threshold, MAX_CHUNKS

def embed_sentence(sentence, vocabulary):
    embedded = [0] * MAX_CHUNKS
    for index, chunk in enumerate(sentence.chunks[:MAX_CHUNKS]):
        word = chunk.clean_name
        if word in vocabulary:
            word_id = vocabulary[word]
            embedded[index] = word_id
        else:
            embedded[index] = len(vocab)+1
    return embedded

def apply_commas(sentence, commas):
    assert len(sentence.chunks) == len(commas)
    temp = ""
    for chunk, comma in zip(sentence.chunks, commas):
        if comma == chunk.comma:
            temp += chunk.name + chunk.trailing_whitespace
            continue
        
        if comma:
            temp += chunk.name + "," + chunk.trailing_whitespace
        else:
            temp += chunk.name[:-1] + chunk.trailing_whitespace
    return temp

def manual_error_analysis(eu_data_dev, idx, Y, Y_hat, threshold=0.5):
    if idx is None:
        agree = np.array([np.array_equal(y, y_hat>threshold) for y, y_hat in zip(Y, Y_hat)])
        idx_of_wrong = np.where(agree==0)[0]
        idx = idx_of_wrong[np.random.randint(len(idx_of_wrong))]
    y = Y[idx]
    y_hat = Y_hat[idx]
    words = eu_data_dev[idx].features
    idx_true = np.where(y==1)[0]
    idx_pred = np.where(y_hat>threshold)[0]
    for true_idx in idx_true:
        if true_idx in idx_pred:
            words[true_idx] = words[true_idx]+', [TP]'
        else:
            words[true_idx] = words[true_idx]+', [FN]'
    for pred_idx in idx_pred:
        if pred_idx not in idx_true:
            words[pred_idx] = words[pred_idx]+' [FP]'
    error_sentence = ' '.join(words)
    return error_sentence

def manual_valid_analysis(eu_data_dev, idx, Y, Y_hat, threshold=0.5):
    if idx is None:
        agree = np.array([np.array_equal(y, y_hat>threshold) for y, y_hat in zip(Y, Y_hat)])
        idx_of_wrong = np.where(agree==1)[0]
        idx = idx_of_wrong[np.random.randint(len(idx_of_wrong))]
    y = Y[idx]
    y_hat = Y_hat[idx]
    words = eu_data_dev[idx].features
    idx_true = np.where(y==1)[0]
    idx_pred = np.where(y_hat>threshold)[0]
    for true_idx in idx_true:
        if true_idx in idx_pred:
            words[true_idx] = words[true_idx]+', [TP]'
        else:
            words[true_idx] = words[true_idx]+', [FN]'
    for pred_idx in idx_pred:
        if pred_idx not in idx_true:
            words[pred_idx] = words[pred_idx]+' [FP]'
    valid_sentence = ' '.join(words)
    return valid_sentence

#----------------------------------------------------------------------------------------
# VIEWS
#----------------------------------------------------------------------------------------
eu_data_dev = np.load('models/eu_data_dev.npy')
Y_dev = np.load('models/Y_dev.npy')
Y_hat = np.load('models/Y_hat.npy')
vocab, model, threshold, MAX_CHUNKS = initialize_models()

@app.route('/', methods=['POST', 'GET'])
def landing_page():
    print('Site ready to be used')
    text = 'test'
    form = NameForm()
    if form.text.data is not None:
        parser = StringParser()
        text_raw = form.text.data
        text = parser.parse(text_raw)
        embedded = embed_sentence(text, vocab)
        embedded = sequence.pad_sequences([embedded], maxlen=MAX_CHUNKS)
        yhat = model.predict_proba(embedded)
        y_hat_commas = yhat[0][:len(text.chunks)] >= threshold
        processed = apply_commas(text, y_hat_commas)
    else:
        print('form not valid')
        processed = ''
    return render_template('recommendations.html',form=form, text=processed)

@app.route('/erroranalysis', methods=['POST', 'GET'])
def manuel_error_analysis():
    idx = None
    threshold = 0.39
    error_example = manual_error_analysis(eu_data_dev, idx, Y_dev, Y_hat, threshold=threshold)
    return render_template('erroranalysis.html', text=error_example)

@app.route('/validanalysis', methods=['POST', 'GET'])
def manuel_valid_analysis():
    idx = None
    threshold = 0.39
    valid_example = manual_valid_analysis(eu_data_dev, idx, Y_dev, Y_hat, threshold=threshold)
    return render_template('validanalysis.html', text=valid_example)

if __name__ == '__main__':
    manager.run()