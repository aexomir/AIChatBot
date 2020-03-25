# Creating Function for compiling and preprocessing Text
# and Using Model to Predict the result

# This is one of the project i've redesigned..
# You can find the original project in GeeksForGeeks site...
# if you have any problem running or understanding this project,
# feel free to contact me...


# imports
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

model = load_model('')
intents = json.loads(open('dataset/intents.json','r').read())
words = pickle.load(open('dataset/words.pkl','rb'))
classes = pickle.load(open('dataset/classes.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize each sentence to words
    sentence_words = nltk.word_tokenize(sentence)
    # lemmatizer for array(sentence.words)
    
    sentence_words = [WordNetLemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bow(sentence,words,show_details=True):
    sentence_words = clean_up_sentence(sentence=sentence)
    # bag of words
    bag = [0] * len(words)                          ## in words ro nafahmidam chie!
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if the current word is in the vocabulary's position..
                bag[i] = 1
                if show_details:
                    print(f'found in bag {w}')
    return np.array(bag)

def predict_class(sentence,model):
    # filter out predictions below a threshold
    pr = bow(sentence,words,show_details=True)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.1
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]            #inam khub nafahmidam

    # sort by probability
    results.sort(key=lambda x : x[1] , reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]],"probability": str(r[1])})

    return return_list

def get_response(ints,intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['response'])
            break

    return result

def chatBot_response(text):
    ints = predict_class(text,model)
    result = get_response(ints,intents)
    return result



