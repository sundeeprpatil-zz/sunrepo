Step 1: Data preparation
Take text, create corpus of sentences
Encode the sentences with count of alphabet positions
"aaabbbcc" will be encoded as [3,3,2,0...]
X = Scale the encoded matrix
Y = 2 classes : {0,1} are one-hot encoded using keras.utils.to_categorical
---------------------------------------------------------------------
Step 2: Training, Validation and Test set
Use the sklearn utils to split training and test data (80/20)
train_test_split(X, Y, test_size=0.20, random_state=seed)
Training data:  6089 sentences
Test data: 1523 sentences
----------------------------------------------------------------------
Step 3: Setting up the model parameters

3 layer neural network with 50% dropout
Layer 1: 500 neurons with 50% dropout
Layer 2: 300 neurons with 50% dropout
Layer 3: 100 neurons with 50% dropout

Total params: 212,602
Trainable params: 212,602

we used adam optimizer with learning rate: 0.001 and 12 epochs chosen to train the mode. 12 times we iterate over the entire x and y data provided. 10% of fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling.

---------------------------------------------------------------------------
Step 4: Train the model on training data and tune parameters with validation set
model.fit on shuffled training data with batch size 64 and 12 epochs and
10% of the training data to be used as validation data.


---------------------------------------------------------------------------
Step 5: Evaluate model on unseen test data and check for loss/ accuracy metric
Check also the classification report
Check for the plots on how the evaluation happened so far


---------------------------------------------------------------------------

Step 6: After finalizing the algorithm --> provide new data as needed and run the model to get the predicted language



---------------------------------------------------------------------------







 'abcdefghijklmnopqrstuvwxyzßäöü !?#]ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ')
