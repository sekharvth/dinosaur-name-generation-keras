# Generate made up dinosuar names

Dinosuar names tend to follow a particualar pattern, with most ending in '-saurus'. Here, the data containing many dinosaur is available.
We use a character level language generation model trained on this data to generate made up dinosaur names. 

The input is a sequence of one-hot representation of each character in the name. The output is the same name as the input, but shifted one time
step to the left. It ends with an 'EOS' tag. So the lengths of input and output are the same.
