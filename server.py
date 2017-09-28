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
manager.add_command("local", Server(host="0.0.0.0", port=5000))
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

vocab, model, threshold, MAX_CHUNKS = initialize_models()

print('Do I even get here?')

#----------------------------------------------------------------------------------------
# VIEWS
#----------------------------------------------------------------------------------------

@app.route('/', methods=['POST', 'GET'])
def hello_world():
    text = 'test'
    form = NameForm()
    if form.text.data is not None:
        parser = StringParser()
        text_raw = form.text.data
        print('Raw text is:', text_raw)
        text = parser.parse(text_raw)
        embedded = embed_sentence(text, vocab)
        embedded = sequence.pad_sequences([embedded], maxlen=MAX_CHUNKS)
        yhat = model.predict_proba(embedded)
        print('yhat:', yhat)
        y_hat_commas = yhat[0][:len(text.chunks)] >= threshold
        print('y_hat_commas:', y_hat_commas)
        processed = apply_commas(text, y_hat_commas)
        print('processed', processed)
    else:
        print('form not valid')
        processed = ''
    return render_template('recommendations.html',form=form, text=processed)

if __name__ == '__main__':
    manager.run()