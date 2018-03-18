# The code that follows is vaguely similar to the model in the seq2seq implementation of the simple chatbot I developed, which can be 
# found here: 

# import necessary packages
import numpy as np
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K


# X is naturally, the input data, and of shape (num_examples, max_length_of_sentence, vocab_size)
# Y is the output, of shape (amx_length_of_sentence, num_examples, vocab_size). The explanation for this difference in shapes b/w 
# input and output is given later on.
# It doesn't matter if lengths of sentences aren't same throughout the X and Y data set, as we'll be using for loops for looping through
# the words in each sentence. Thus we don't need to pad the sequences.
X, Y, vocab_size = load_data()

# for use later on. In the seq2seq models, we use the embeddings of words as input to the model. But here, we just use the one-hot
# encodings as inputs, which is what the function below does. 
def one_hot(x, vocab_size):
    x = K.argmax(x)
    x = tf.one_hot(x, vocab_size) 
# the RepeatVector statement is used to change x into 3 dimensions. Right now, x is of shape (num_examples, vocab_size).
# But in the inference model defined later on, this x is to be fed as input for the next time step, whose model architecture 
# defines the input to be of shape (1, vocab_size) excluding the num_examples. 
# RepeatVector makes x transform from (num_examples, vocab_size), to (num_examples, 1, vocab_size) making it ready to be used as the 
# next input
    x = RepeatVector(1)(x)
    return x

# the number of units for each LSTM cell
n_a = 300 

# the number of unique words. 
vocab_size = 50,000

# define the layers to be used and make them global
reshapor = Reshape((1, vocab_size))                        
LSTM_cell = LSTM(n_a, return_state = True)        
densor = Dense(vocab_size, activation='softmax')     

# model for training part. 
# Tx refers to the number of words in each sentence of the input, or alternatively, the number of time steps in the input.
def djmodel(Tx, n_a, vocab_size):
    
    # Define the inputs to the model
    X = Input(shape=(Tx, vocab_size))
    # initial hidden states of the LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    # Since the model generates one output per time step, (since we haven't set the return_states argument to True in the definition of 
    # LSTM_cell), we use a list to append the output at each time step, to complete the answer sentence
    outputs = []
    
    # loop through the words in the input sentence
    for t in range(Tx):
        # pick out the one-hot representation of the 't'th word in the sentence, using a Lambda function
        x = Lambda(lambda x: x[:, t, :])(X)
        
        # reshape it to be fit for input for the LSTM
        x = reshapor(x)
       
        # pass the input to the LSTM cell, with the initial hidden states. Then re-initialize the hidden states for the next
        # time step with the output states of the current time step
        a, _, c = LSTM_cell(x, initial_state = [a, c])
        
        # pass the output of the LSTM into the densor layer, that gives the softmax activation with 'vocab_size' units
        out = densor(a)
        
        # append the output to the outputs list. Now, 'out' is of shape (num_examples, vocab_size).
        # When appending it to the list 'outputs', ultimately the shape becomes (Tx, num_examples, vocab_size) and that is 
        # why the actual target outputs have been made to be the shape of 'outputs'.
        outputs.append(out)
        
    # define the model instance, with the one-hot encoded input sentences and initial states to the LSTM as input, 
    # and one-hot representations of the actual target sentences as outputs.
    model = Model(inputs = [X, a0, c0], outputs = outputs)
    
    return model

# call the djmodel function and assign it to 'model'
model = djmodel(Tx = 30 , n_a = 300, vocab_size)

# define the optimizer
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


m = num_examples
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

model.fit([X, a0, c0], list(Y), epochs=100)


def inference_model_1(LSTM_cell, densor, vocab_size = 78, n_a = 64, Ty = 100):
   
   
    x0 = Input(shape=(1, vocab_size))
    
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    outputs = []
    
    for t in range(Ty):
        
        a, _, c = LSTM_cell(x, initial_state = [a,c])
        
        out = densor(a)
    
        outputs.append(out)
        
        x = Lambda(one_hot)(out)
        
    inference_model = Model(inputs = [x0, a0, c0], outputs = outputs)
    
    return inference_model


inference_model = inference_model_1(LSTM_cell, densor, vocab_size, n_a = 64, Ty = 50)


x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    
    indices = np.argmax(pred, axis = -1)
  
    results = to_categorical(indices)
   
    
    return results, indices

results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)





