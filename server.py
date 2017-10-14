import os
import numpy as np
from flask import Flask
from flask import request, render_template, session
from flask_script import Manager, Shell, Server
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField
import src
from komma_dev.parsing import StringParser
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
    submit = SubmitField('Sæt kommaer')
#----------------------------------------------------------------------------------------
# Initializing models and parameters
#----------------------------------------------------------------------------------------
eu_data_dev, Y_dev, Y_hat = src.loading_data()
vocab, model, threshold, MAX_CHUNKS = src.initialize_models()
parser = StringParser()
threshold = 0.39
print('Site ready to be used')
#----------------------------------------------------------------------------------------
# VIEWS
#----------------------------------------------------------------------------------------
@app.route('/', methods=['POST', 'GET'])
def landing_page():
    form = NameForm()
    text_input = form.text.data
    if text_input is not None:
        processed, proba = src.process_input(text_input, parser, vocab, MAX_CHUNKS, model, threshold)
    else:
        print('form not valid')
        processed, proba = '', [0]
    return render_template('recommendations.html',form=form, text=processed, proba=proba)

@app.route('/erroranalysis', methods=['POST', 'GET'])
def manuel_error_analysis():
    error_example, printable_proba, fn_idx, fp_idx, tp_idx = src.manual_error_analysis(
        eu_data_dev,
        None,
        Y_dev,
        Y_hat,
        threshold=threshold
        )
    return render_template(
                            'erroranalysis.html',
                            text=error_example,
                            proba=printable_proba,
                            fp=fp_idx, fn=fn_idx, tp=tp_idx
                            )

@app.route('/validanalysis', methods=['POST', 'GET'])
def manuel_valid_analysis():
    valid_example, printable_proba, tp_idx = src.manual_valid_analysis(eu_data_dev, None, Y_dev, Y_hat, threshold=threshold)
    return render_template(
                            'validanalysis.html',
                            text=valid_example,
                            proba=printable_proba,
                            tp=tp_idx
                            )

if __name__ == '__main__':
    manager.run()