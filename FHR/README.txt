Platform/System: Ubuntu 17.10 64-bit
Python = 3.6.3
Tensorflow/Tensorboard = 1.8.0
numpy = 1.14.5
pandas = 0.23.1


Instructions: 

Code can be run from top down. However, two variables are going to need to be set.
I had to run this code within a linux vm as TensorRandomForest has ops/kernals that are not supported
on windows. I'm not an expert in linux file system so I had to hard code the directory. But it should point to
the location you are running the code.

----===Lines 14 & 15===----
maindir = '/home/claude/Documents/DS_CERT/'

df = pd.read_csv(maindir +'input/train2.csv')
