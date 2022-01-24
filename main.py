# ==============Basic Libraries===========================#
import numpy as np
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


# ============Deep Learning Libraries====================#
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model


# ===========Miscellaneous Files==========================#
import random
import json
import pickle
import tflearn

# ===========Kivy Tools===================================#
import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.utils import get_color_from_hex
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from termcolor import colored


intents_file = open('intents.json').read()
intents = json.loads(intents_file)
print(intents)

words = []
labels = []
docs_x = []
docs_y = []
ignore_letters = ['!', '?', ',', '.']

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        wrds = word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Training with Deep Neural Network

model = Sequential()
# first layer
model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
# second layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# third layer - > output with layer size of 14
model.add(Dense(len(output[0]), activation='softmax'))

# Compiling model...

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Training and saving the model
hist = model.fit(training, output, epochs=1000, batch_size=32, verbose=1)

print("model is created")

# Saving the results of the model and other documents for predictions later on.

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

model.save('chatbot_model.h5', hist)

# ===========================================================================================================#

# For defining the color of the background
Window.clearcolor = get_color_from_hex('#FFFFFF')


# This will represent main window
class MainWindow(Screen):
    def quit(self, obj):
        print(obj)

        App.get_application_icon(self).stop()
        Window.close()


Window.size = (400, 600)
kv = Builder.load_file("main.kv")


# ===============NLP pre-processing part ================================================#
# Opening the trained model
model = keras.models.load_model('chatbot_model.h5')
# Opening the pickle file for vocabulary and encoders
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


# function to start the chat
def predict_label(sentence, label, intents_json):
    # The response we get is a probability distribution of nodes (softmax)
    # We should filter the prediction thresholds to keep the largest.
    b = bag_of_words(sentence, words)

    results = model.predict(np.array([b]))[0] # This means pick the first list. (we have list of lists as output)
    result_index = np.argmax(results) # This returns the index of the largest value in the predicted results
    tag = label[result_index] # Labels stores all our labels and the index of the

    if results[result_index] > 0.7:

        for tg in intents_json['intents']:
            if tg['tag'] == tag:
                response = random.choice(tg['responses'])
                link = random.choice(tg['link'])
                result = response + "\n" + link
                break

        return result
    else:

        return "I'm sorry, I don't understand you!"


# This class will represent the chat-bot application!

class ChatWindow(Screen):
    message = ObjectProperty(None)
    main = ObjectProperty(None)
    history = ObjectProperty(None)


    def send_message(self):
        person = "You: "
        bot = "Bot: "
        msg = self.message.text
        self.reset()
        self.history.update_chat_history(f"[color=dd2020]{person}[/color] : {msg}")
        result = predict_label(msg, labels, intents)
        self.history.update_chat_history(f"[color=18dc22]{bot}[/color] : {result}")

    def reset(self):
        self.message.text = ""

# We are creating a scrollable label that we can use to scroll the text.
# We create two widgets: One to scroll as new messages come in,
class ScrollableLabel(ScrollView):
    chat_hist = ObjectProperty(None)
    layout = ObjectProperty(None)
    scroll = ObjectProperty(None)

    def update_chat_history(self, message):
        # We add the message
        self.chat_hist.text += '\n' + message
        # We get the height and add that information and then set the text and size.
        self.layout.height = self.chat_hist.texture_size[1] + 20
        self.chat_hist.height = self.chat_hist.texture_size[1]

        self.chat_hist.text_size = (self.chat_hist.width*0.99, None)

        self.scroll_to(self.scroll)


class ChatbotApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MainWindow(name='main'))
        sm.add_widget(ChatWindow(name='chat'))
        return sm

if __name__ == '__main__':
    ChatbotApp().run()
