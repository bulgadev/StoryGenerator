import numpy as np
#tensorFlow to train it
import tensorflow as tf
#sequential to build the model
from tensorflow.keras.models import Sequential # type: ignore
#LTSM to understand the hihstory, dense for prediction, activation more prediction
from tensorflow.keras.layers import LSTM, Dense, Activation # type: ignore
#makes the ai learn faster
from tensorflow.keras.optimizers import Adam # type: ignore

#exemplos de frases pra ia
corpus = [
    "The friends found a old map below the bed.",
    "And he went to adventure the world step by step.",
    "As he arrived he saw a beatiful island.",
    "Knowledge is power."
    "And they build a simple cart."
]

#this turns word into numbers cuz ai need numbers not words
#list with word to numbers
word_to_int = {}
#List with number to words
int_to_word = {}
index = 0
#gets each setence from our examples
for setence in corpus:
    #separe it into words
    for word in setence.split():
        #transform into numbers and add it to our list
        if word not in word_to_int:
            word_to_int[word] = index
            int_to_word[index] = word
            index += 1

#so this turns each word into a list of numbers to use as training data
# ['the', 'friends', ' found'] => [0, 1, 2]
sequences = []
for setence in corpus:
    #adds to the list the word transformed into numbers
    sequences.append([word_to_int[word] for word in setence.split()])

#this make the ai learn wuts the next thing to guess
#so The friends found a -> old, map
#for ai is something like, 0,1,3, and it guesses 4
#how many words it gets before nedding to guess
sequences_length = 3
#x is the inputs (wut ai sees)
#y is answers(wut ai has to guess)
x = []
y = []

#loops in each of our sequences list
for seq in sequences:
    #separe it on 3word slices
    #i in range of the length of seq it gets 3
    for i in range(len(seq) - sequences_length):
        x.append(seq[i:i+sequences_length]) #adds the input to our x list
        y.append(seq[i+sequences_length]) #adds the output to the y list

#we transform our x and y lists, our data, into numpy arrays
x = np.array(x)
y = np.array(y)

#makes it on a np array that tesorflow understands
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

#build the model
model = Sequential()
#The memory of our ai, using lstm, it makes the memory theh sequence length
model.add(LSTM(50, input_shape=(sequences_length, 1), return_sequences = True))
model.add(LSTM(50))
#the dense to predict the word
model.add(Dense(len(word_to_int)))
#picks the most likely word
model.add(Activation('softmax'))

#training the model
#look coll using sequential, dont it?
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy')
#10 times = 10 epochs, so it read my setences 10 times
#batch size means that it trains on example at a time
model.fit(x, y, epochs=1200, batch_size=1)

#func to make the ai tell the story
def generate_text(seed_setence, length=10):
    #gives a starting setence, keeps doing it and then a story is build
    words = seed_setence.split()
    #training yippie
    for _ in range(length):
        input_sequence = [word_to_int[word] for word in words[-sequences_length:]]
        input_sequence = np.reshape(input_sequence, (1, len(input_sequence), 1))

        predicted = model.predict(input_sequence)
        predicted_word = int_to_word[np.argmax(predicted)]

        words.append(predicted_word)
    return ' '.join(words)

#test it
seed_setence = "The friends found"
generated_text = generate_text(seed_setence)
print(generated_text)