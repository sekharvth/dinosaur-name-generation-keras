# Generate made up dinosuar names

This was an assignment I had to do as part of a course in Deep Learning on Coursera. The original assignment had to be done without the use of any programming frameworks. This version uses Keras for training and inference. 

Dinosuar names tend to follow a particualar pattern, with most ending in '-saurus'. Here, the data containing many dinosaur is available.
We use a character level language generation model trained on this data to generate made up dinosaur names. 

The input is a sequence of one-hot representation of each character in the name. The output is the same name as the input, but shifted one time step to the left. It ends with an 'EOS' tag. So the lengths of input and output are the same.

Instead of using vectorization during the training process, a for loop is used to run through the time steps of each individual training 
example. As a result, it uses functions like Lambda and RepeatVector through Keras.

At each time step during training, the input character to the LSTM is mapped to the next character in the sequence, and the last character is mapped to the EOS tag. At each step, the previous hidden states of the LSTM are also passed to the LSTM along with the input char.

During inference, a character chosen by the user is input into the trained layers, and it iterates till the 'EOS' tag is generated.
