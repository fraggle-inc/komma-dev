import numpy as np 
from keras.models import load_model
from komma_dev.parsing import StringParser
from keras.preprocessing import sequence

def initialize_models():
    print('Loading the vocabulary and model')
    vocab = np.load('models/vocab_50k.npy').item()
    model = load_model('models/model_50k_vocab.h5')
    threshold = 0.39
    MAX_CHUNKS = 50
    return vocab, model, threshold, MAX_CHUNKS

def loading_data():
    print('Loading the data')
    eu_data_dev = np.load('models/eu_data_dev.npy')
    Y_dev = np.load('models/Y_dev.npy')
    Y_hat = np.load('models/Y_hat.npy')
    return eu_data_dev, Y_dev, Y_hat

def embed_sentence(sentence, vocabulary, MAX_CHUNKS):
    embedded = [0] * MAX_CHUNKS
    for index, chunk in enumerate(sentence.chunks[:MAX_CHUNKS]):
        word = chunk.clean_name
        if word in vocabulary:
            word_id = vocabulary[word]
            embedded[index] = word_id
        else:
            embedded[index] = len(vocabulary)+1
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

def apply_commas_to_chuncks(sentence, commas):
    assert len(sentence.chunks) == len(commas)
    temp = []
    for chunk, comma in zip(sentence.chunks, commas):
        if comma == chunk.comma:
            temp.append(chunk.name + chunk.trailing_whitespace)
            continue
        
        if comma:
            temp.append(chunk.name + "," + chunk.trailing_whitespace)
        else:
            temp.append(chunk.name[:-1] + chunk.trailing_whitespace)
    return temp

def process_input(text_input, parser, vocab, MAX_CHUNKS, model, threshold):
    text = parser.parse(text_input)
    embedded = embed_sentence(text, vocab, MAX_CHUNKS)
    embedded = sequence.pad_sequences([embedded], maxlen=MAX_CHUNKS)
    yhat = model.predict_proba(embedded)
    y_hat_commas = yhat[0][:len(text.chunks)] >= threshold
    printable_proba = [int(proba*100)/100 for proba in yhat[0][:len(text.chunks)]]
    printable_text = apply_commas_to_chuncks(text, y_hat_commas)
    return printable_text, printable_proba

def manual_error_analysis(eu_data_dev, idx, Y, Y_hat, threshold=0.5):
    if idx is None:
        agree = np.array([np.array_equal(y, y_hat>threshold) for y, y_hat in zip(Y, Y_hat)])
        idx_of_wrong = np.where(agree==0)[0]
        idx = idx_of_wrong[np.random.randint(len(idx_of_wrong))]
    y = Y[idx]
    y_hat = Y_hat[idx]
    words = eu_data_dev[idx].features
    printable_proba = [int(proba*100)/100 for proba in y_hat[:len(eu_data_dev[idx].chunks)]]

    idx_true = np.where(y==1)[0]
    idx_pred = np.where(y_hat>threshold)[0]
    fn_idx, fp_idx, tp_idx = [], [], []
    
    for true_idx in idx_true:
        if true_idx in idx_pred:
            words[true_idx] = words[true_idx]+','
            tp_idx.append(true_idx)
        else:
            words[true_idx] = words[true_idx]+','
            fn_idx.append(true_idx)
    for pred_idx in idx_pred:
        if pred_idx not in idx_true:
            words[pred_idx] = words[pred_idx]
            fp_idx.append(pred_idx)
    return words, printable_proba, fn_idx, fp_idx, tp_idx

def manual_valid_analysis(eu_data_dev, idx, Y, Y_hat, threshold=0.5):
    if idx is None:
        agree = np.array([np.array_equal(y, y_hat>threshold) for y, y_hat in zip(Y, Y_hat)])
        idx_of_wrong = np.where(agree==1)[0]
        idx = idx_of_wrong[np.random.randint(len(idx_of_wrong))]
    y = Y[idx]
    y_hat = Y_hat[idx]
    printable_proba = [int(proba*100)/100 for proba in y_hat[:len(eu_data_dev[idx].chunks)]]
    words = eu_data_dev[idx].features

    idx_true = np.where(y==1)[0]
    idx_pred = np.where(y_hat>threshold)[0]
    tp_idx = []
    for true_idx in idx_true:
        if true_idx in idx_pred:
            words[true_idx] = words[true_idx]+','
            tp_idx.append(true_idx)
        else:
            words[true_idx] = words[true_idx]+','
    for pred_idx in idx_pred:
        if pred_idx not in idx_true:
            words[pred_idx] = words[pred_idx]
    return words, printable_proba, tp_idx