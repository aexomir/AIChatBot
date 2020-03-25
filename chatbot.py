# Creating Deep Learning Model using Tensorflow and Keras
# tf : used for training data
# keras: used for Sequential network model creation

# This is one of the project i've redesigned..
# You can find the original project in GeeksForGeeks site...
# if you have any problem running or understanding this project,
# feel free to contact me...





#### import requirements ###
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout
from keras.optimizers import SGD
import random

# load data files
words = []
classes = []
documents = []
ignore_words = ['?','!']
data_file = open("dataset/intents.json",'r').read()
intents = json.loads(data_file)

### PreProcess Data ###

# putting all words of all patterns
# in intents in a list "documents"
# and putting all class tags in a list "classes"
# p.s: tokenized words in a list "words"
for intent in intents['intent']:
    for pattern in intents['pattern']:
        # tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #add documents in the corpus
        documents.append((word,intent['tag']))
        # adding classes by tags
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize , lower each and remove duplicates
        words = [WordNetLemmatizer.lemmatize(word.lower()) for words in words if word not in ignore_words]
        word = sorted(list(set(words)))
        classes = sorted(list(set(classes)))
        print(f"{len(words)} words,{len(documents)} documents,{len(classes)} classes")
        pickle.dump(words,open('dataset/words.pkl','wb'))
        pickle.dump(classes,open("dataset/classes.pkl","wb"))




print(documents)
### Training Data ###

# we create the training data
# input : pattern
# output : class of input
# cp doesn't know text ; so we
# convert to numbers

# create training data
training = []
number_output = [0] * len(classes)

for doc in documents:
    # initialize tokenized words in doc using "bag"
    bag = []
    pattern_words = doc[0]
    pattern_words = [WordNetLemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1 , if word match found in current pattern
    for word in words:
        if word in pattern_words:
            bag.append(1)
    else:
        bag.append(0)
    # output is 1 for the current tag and others are 0
    output_row = list(number_output)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag,output_row])

# Shuffle features and put into np.array
random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])
print('Training Data Created..')


### Building Model ###

# model chatbot_model h5
# 128 >> 64 >> len(classes)
model = Sequential()
model.add(Dense(128,input_shape=len(train_x[0]),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

# Compile model.
# Stochastic gradient descent with
# Nesterov accelerated gradient
# gives good results for this model
sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical-crossentropy',optimizer=sgd,metrics=['accuracy'])

hist = model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('chatbot_model.h5',hist)
print('Model Created..')

