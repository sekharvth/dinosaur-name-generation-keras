
import numpy as np
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K



X, Y, n_values, indices_values = load_music_utils()
print('shape of X:', X.shape)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)

def one_hot(x):
    x = K.argmax(x)
    x = tf.one_hot(x, 78) 
    x = RepeatVector(1)(x)
    return x

n_a = 64 

reshapor = Reshape((1, 78))                        
LSTM_cell = LSTM(n_a, return_state = True)        
densor = Dense(n_values, activation='softmax')     

def djmodel(Tx, n_a, n_values):
  
    X = Input(shape=(Tx, n_values))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    outputs = []
    
    for t in range(Tx):
        
        x = Lambda(lambda x: x[:, t, :])(X)
        
        x = reshapor(x)
       
        a, _, c = LSTM_cell(x, initial_state = [a, c])
        
        out = densor(a)
        
        outputs.append(out)
        
 
    model = Model(inputs = [X, a0, c0], outputs = outputs)
    
    return model


model = djmodel(Tx = 30 , n_a = 64, n_values = 78)

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

model.fit([X, a0, c0], list(Y), epochs=100)


def inference_model_1(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100):
   
   
    x0 = Input(shape=(1, n_values))
    
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


inference_model = inference_model_1(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)


x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    
    indices = np.argmax(pred, axis = -1)
    # Step 3: Convert indices to one-hot vectors, the shape of the results should be (1, )
    results = to_categorical(indices)
    ### END CODE HERE ###
    
    return results, indices


# In[23]:

results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))





