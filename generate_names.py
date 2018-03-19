# The code that follows is vaguely similar to the model in the seq2seq implementation of the simple chatbot I developed, which can be 
# found here: https://github.com/sekharvth/simple-chatbot-keras

# import necessary packages
import numpy as np
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K

# load data
context = np.load('dino_names_char_level.npy')
final_target = np.load('dino_names_char_level.npy')
with open('dictionary_char_level.pkl', 'rb') as f:
    char_to_index = pickle.load(f)
with open('reverse_dictionary_char_level.pkl', 'rb') as f:
    index_to_char = pickle.load(f)

X, Y, vocab_size = context, final_target, len(char_to_index)
# X is naturally, the input data(dinosaur names). It is of shape (num_examples, dino_name_length, vocab_size). The name lenghts can be
# different in each example, as we use a for loop here, and no padding (which makes all examples same length) is required.
# Y is the same as X, but shifted on time step to the left, and ends with a 'EOS' tag. 
# It is of shape (dino_name_length, num_examples, vocab_size). 
# The explanation for this difference in shapes b/w input and output is given later on.

# for use later on during the inference mode. Here, we just use the one-hot encodings as inputs, which is what the function below does. 
def one_hot(x, vocab_size):
    x = np.random.choice(a = vocab_size, p = x.ravel())
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

# define the layers to be used and make them global
reshapor = Reshape((1, vocab_size))                        
LSTM_cell = LSTM(n_a, return_state = True)        
densor = Dense(vocab_size, activation='softmax')     

# model for training part. 
# Tx refers to the number of characters in each name in the input, or alternatively, the number of time steps in the input.
def djmodel(Tx, n_a, vocab_size):
    
    # Define the inputs to the model
    X = Input(shape=(Tx, vocab_size))
    
    # initial hidden states of the LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    # Since the model generates one output per time step, (as we haven't set the return_states argument to True in the definition of 
    # LSTM_cell), we use a list to append the output at each time step, to complete the answer sentence
    outputs = []
    
    # loop through the characters in the input name
    for t in range(Tx):
        # pick out the one-hot representation of the 't'th character in the name, using a Lambda function
        x = Lambda(lambda x: x[:, t, :])(X)
        
        # reshape it to be fit for input to the LSTM
        x = reshapor(x)
       
        # pass the input to the LSTM cell, with the initial hidden states. Then re-initialize the hidden states for the next
        # time step with the output states of the current time step
        a, _, c = LSTM_cell(x, initial_state = [a, c])
        
        # pass the output of the LSTM into the densor layer, that gives the softmax activation with 'vocab_size' units
        out = densor(a)
        
        # append the output to the outputs list. Now, 'out' is of shape (num_examples, vocab_size).
        # When appending it to the list 'outputs', ultimately the shape becomes (Tx, num_examples, vocab_size) and that is 
        # why the actual target outputs have been made to be of shape (dino_name_length, num_examples, vocab_size)
        outputs.append(out)
        
    # define the model instance, with the one-hot encoded input names and initial states to the LSTM as input, 
    # and one-hot representations of the actual target names as outputs.
    model = Model(inputs = [X, a0, c0], outputs = outputs)
    
    return model

# call the djmodel function and assign it to 'model'
model = djmodel(Tx , n_a = 300, vocab_size)

# define the optimizer and compile the model
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# initialise variables to feed into the model as initial input
m = num_examples
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

# fit the model. Since the output is a list, we convert Y into a list too
model.fit([X, a0, c0], list(Y), epochs=50000)

# define the function for the inference model
def inference_model_1(LSTM_cell, densor, vocab_size, n_a):
   
    # define the inputs to the model
    x0 = Input(shape=(1, vocab_size))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0
    
    # empty list to store the outputs
    outputs = []
    
    # loop over the characters in the target name (max sequence length)
    while (K.argmax(out) != index_to_char['EOS']):
        
        # pass the input to the LSTM cell(trained previoulsy, as it is a global layer), with the initial hidden states. 
        # Then re-initialize the hidden states for the next time step with the output states of the current time step
        a, _, c = LSTM_cell(x, initial_state = [a,c])
        
        # output of the current time step
        out = densor(a)
        
        # append to list
        outputs.append(out)
        
        # call the one_hot() function previously defined, with the 'softmax' activated dense layer (shape = m, vocab_size) as input
        x = Lambda(one_hot)(out)
    
    # create the model instance
    inference_model = Model(inputs = [x0, a0, c0], outputs = outputs)
    
    return inference_model


inference_model = inference_model_1(LSTM_cell, densor, vocab_size, n_a)

# initialise variables to pass as input
x_initialiser = np.zeros((1, 1, vocab_size))
a_initialiser = np.zeros((1, n_a))
c_initialiser = np.zeros((1, n_a))

# generate dinosaur names starting with 's'
x_initialiser[0,0, char_to_index['s']] = 1

def predict_and_sample(inference_model, x_initialiser, a_initialiser, c_initialiser):
    
    # predict the output sequence
    pred = inference_model.predict([x_initialiser, a_initialiser, c_initialiser])
    
    # find the indexes of characters that have the greatest probability in each time steps output
    indices = np.argmax(pred, axis = -1)
   
    return indices

indices = predict_and_sample(inference_model, x_initialiser, a_initialiser, c_initialiser)

# so that the shape changes from (dino_name_length, num_examples) to (num_examples, dino_name_length)
indices = indices.swapaxes(0,1)

# join the characters in the output to form a legible answer
answer = ''.join([index_to_char[i] for i in indices[0]])

