###
#Step 1:
languages_dict = {'en':0,'de':1} # covers English and German language detector
text_sample_size = 140 #140 chars as in sms
num_lang_samples = 3806 # min. samples from dictionary

#Step 2:



# imports
import os
import re

import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras.optimizers


'''
def corpus_sentence_tokens(corpus_text_file):
    while True:
        with open(corpus_text_file, encoding='utf-8') as f:
            for line in f.readlines():
                n,l = line.split('\t')   # Strip of the initial numbers
                for s in sentence_splitter.tokenize(l):  # Split the lines into sentences (~1 each)
                    tree_banked = tokenizer.tokenize(s)
                    if len(tree_banked) < text_sample_size:
                        yield tree_banked
        print("Corpus : Looping")

corpus_sentence_tokens_gen = corpus_sentence_tokens()
'''


# output:  small, big characters, letters =[small, big, special]
def define_alphabet():
    base_en = 'abcdefghijklmnopqrstuvwxyz'
    special_chars = ' !?#'
    german = 'äöüß'
    all_lang_chars = base_en + german
    small_chars = list(set(list(all_lang_chars)))
    small_chars.sort()
    big_chars = list(set(list(all_lang_chars.upper())))
    big_chars.sort()
    small_chars += special_chars
    return small_chars,big_chars

alphabet = define_alphabet()
print (alphabet)


def size_mb(size):
    size_mb =  '{:.2f}'.format(size/(1000*1000.0))
    return size_mb + " MB"



# utility function to turn language id into language code
def decode_langid(langid):
    for dname, did in languages_dict.items():
        if did == langid:
            return dname


data_directory = "./Data/"
source_directory = data_directory + 'source' #  dump from wiki
cleaned_directory = data_directory + 'cleaned' # removed spaces, newlines and xml if applicable
samples_directory = data_directory + 'samples' #
train_test_directory = data_directory + 'train_test'

#just summarize the list of files available
for filename in os.listdir(source_directory):
    path = os.path.join(source_directory, filename)
    if not filename.startswith('.'):
        print((path), "size : ",size_mb(os.path.getsize(path)))




for lang_code in languages_dict:
    path_src = os.path.join(source_directory, lang_code+ ".txt")
    f = open(path_src, encoding='utf-8', errors = 'ignore')
    content = f.read()
    print('Language : ',lang_code)
    f.close()
    text = content
    text = re.sub(r'<[^<]+?>', '', text) # suitable for xml (not applicable here)
    text = text.replace('\n', ' ') #new lines
    text = re.sub(r'\s+', ' ', text) # handle multiple spaces
    content = text
    path_cl = os.path.join(cleaned_directory,lang_code + '_cleaned.txt')
    f = open(path_cl,'w', errors = 'ignore')
    f.write(content)
    f.close()
    del content
    print ("Cleaning completed for : " + path_src,'->',path_cl)
    print (100*'-')
print ("END OF CLEANING")





# this function will get sample of text from each cleaned language file.
# It will try to preserve complete words - if word is to be sliced, sample will be shortened to full word
def get_sample_text(file_content, start_index, sample_size):
    # we want to start from full first word
    # if the first character is not space, move to next ones
    while not (file_content[start_index].isspace()):
        start_index += 1
    # now we look for first non-space character - beginning of any word
    while file_content[start_index].isspace():
        start_index += 1
    end_index = start_index + sample_size
    # we also want full words at the end
    while not (file_content[end_index].isspace()):
        end_index -= 1
    return file_content[start_index:end_index]


# we need only alpha characters and some (very limited) special characters
# exactly the ones defined in the alphabet
# no numbers, most of special characters also bring no value for our classification task
# (like dot or comma - they are the same in all of our languages so does not bring additional informational value)

# count number of chars in text based on given alphabet
def count_chars(text, alphabet):
    alphabet_counts = []
    for letter in alphabet:
        count = text.count(letter)
        alphabet_counts.append(count)
    return alphabet_counts


#every sentence gets encoded with smallchars and big chars
def get_input_row(content, start_index, sample_size):
    sample_text = get_sample_text(content, start_index, sample_size)
    counted_chars_all = count_chars(sample_text.lower(), alphabet[0])
    counted_chars_big = count_chars(sample_text, alphabet[1])
    all_parts = counted_chars_all + counted_chars_big
    return all_parts

'''
# let's see if our processing is returning counts
# last part calculates also input_size for DNN so this code must be run before DNN is trained
path = os.path.join(cleaned_directory, "de_cleaned.txt")
with open(path, 'r') as f:
    content = f.read()
    random_index = random.randrange(0, len(content) - 2 * text_sample_size)
    sample_text = get_sample_text(content, random_index, text_sample_size)
    print("1. Sample text: \n", sample_text)
    print("2. Reference alphabet: \n", alphabet[0], alphabet[1])
    sample_input_row = get_input_row(content, random_index, text_sample_size)
    print("3. Sample_input_row: \n", sample_input_row)
    input_size = len(sample_input_row)
    print("4. Input size : ", input_size)
    del content

'''

# now we have preprocessing utility functions ready. Let's use them to process each cleaned language file
# and turn text data into numerical data samples for our neural network
# prepare numpy array
encoding_size = len(alphabet[0]) + len(alphabet[1])


#output: [encoded sentence with alphabets, language_label]
sample_data = np.empty((num_lang_samples * len(languages_dict), encoding_size + 1), dtype=np.uint16)
lang_seq = 0
jump_reduce = 0.2  # part of characters removed from jump to avoid passing the end of file
for lang_code in languages_dict:
    start_index = 0
    path = os.path.join(cleaned_directory, lang_code + "_cleaned.txt")
    with open(path, 'r') as f:
        print("Processing file : " + path)
        file_content = f.read()
        content_length = len(file_content)
        remaining = content_length - text_sample_size * num_lang_samples
        jump = int(((remaining / num_lang_samples) * 3) / 4)
        print("File size : ", size_mb(content_length), \
              " | # possible samples : ", int(content_length / encoding_size), \
              "| # skip chars : " + str(jump))
        for idx in range(num_lang_samples):
            input_row = get_input_row(file_content, start_index, text_sample_size)
            sample_data[num_lang_samples * lang_seq + idx,] = input_row + [languages_dict[lang_code]]
            start_index += text_sample_size + jump
        del file_content
    lang_seq += 1 #next language
    print("Start encoding the next language in sequence")
    print(100*"-")

# let's randomy shuffle the data
np.random.shuffle(sample_data)
# reference input size
print("Size of encoded alphabets : ", encoding_size)
print(100 * "-")

print("Number of samples encoded: Sentences per language*number of languages ", sample_data.shape)
print("Store encoded data locally")
path_smpl = os.path.join(samples_directory, "lang_samples_" + str(encoding_size) + ".npz")
np.savez_compressed(path_smpl, data=sample_data)
print(path_smpl, "size : ", size_mb(os.path.getsize(path_smpl)))
del sample_data



# now we will review the data  - control check step
path_smpl = os.path.join(samples_directory,"lang_samples_"+str(encoding_size)+".npz")
dt = np.load(path_smpl)['data']
random_index = random.randrange(0,dt.shape[0])
print ("Randomly pick an Encoded sample : \n",dt[random_index,])
print ("Language for the ramdom sample : ",decode_langid(dt[random_index,][encoding_size]))
print ("Encoded Dataset shape :", dt.shape)




# scaling to optimization can converge well
# we need also ensure one-hot econding of target classes for softmax output layer
dt = dt.astype(np.float64)
# X and Y split
X = dt[:,0:encoding_size]
Y = dt[:,encoding_size]
del dt
# random index to check random sample
random_index = random.randrange(0,X.shape[0])
print("Example data before processing:")
print("X : \n", X[random_index,])
print("Y : \n", Y[random_index])
standard_scaler = preprocessing.StandardScaler().fit(X)
X = standard_scaler.transform(X)
print ("X preprocessed shape :", X.shape)
Y = keras.utils.to_categorical(Y, num_classes=len(languages_dict)) # Y one-hot encoding

print("Example data after processing:")
print("X : \n", X[random_index,])
print("Y : \n", Y[random_index])

# train/test split. Static seed to have comparable results for different runs
seed = 10
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)
del X, Y
# wait for memory release again
#time.sleep(120)
# save train/test arrays to file
path_tt = os.path.join(train_test_directory,"train_test_data_"+str(encoding_size)+".npz")
np.savez_compressed(path_tt,X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
print(path_tt, "size : ",size_mb(os.path.getsize(path_tt)))
del X_train,Y_train,X_test,Y_test




# load train data first from file
path_tt = os.path.join(train_test_directory,"train_test_data_"+str(encoding_size)+".npz")
train_test_data = np.load(path_tt)
X_train = train_test_data['X_train']
print ("X_train: ",X_train.shape)
Y_train = train_test_data['Y_train']
print ("Y_train: ",Y_train.shape)
X_test = train_test_data['X_test']
print ("X_test: ",X_test.shape)
Y_test = train_test_data['Y_test']
print ("Y_test: ",Y_test.shape)
del train_test_data




# create model
model = Sequential()
model.add(Dense(500,input_dim=encoding_size,kernel_initializer="glorot_uniform",activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(300,kernel_initializer="glorot_uniform",activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(100,kernel_initializer="glorot_uniform",activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(len(languages_dict),kernel_initializer="glorot_uniform",activation="softmax"))

model.summary()

model_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=model_optimizer,
              metrics=['accuracy'])



# let's fit the data
history = model.fit(X_train,Y_train,
          epochs=12, #Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
          validation_split=0.10, # 10% of data not used for training, but used for calculation of loss
          batch_size=64, # Number of samples per gradient update.
          verbose=2,
          shuffle=True) # shuffle the data before each epoch


# Evaluate model on unseen data (TEST data)
scores = model.evaluate(X_test, Y_test, verbose=1)
print("Loss metric:%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("Accuracy metric:%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))




# and now we will prepare data for scikit-learn classification report
Y_pred = model.predict_classes(X_test)
Y_pred = keras.utils.to_categorical(Y_pred, num_classes=len(languages_dict))

# and run the report
target_names =  list(languages_dict.keys())
print(classification_report(Y_test, Y_pred, target_names=target_names))


# show plot accuracy changes during training
plt.plot(history.history['acc'],'g')
plt.plot(history.history['val_acc'],'r')
plt.title('accuracy across epochs')
plt.ylabel('accuracy level')
plt.xlabel('# epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()


# show plot of loss changes during training
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


##Test
en_text = "My name is Sundeep and this is my first NLP project"
de_text = "Mein name ist sundeep und es ist mein erste Deep Learning Projekte"


text_texts_array = [en_text,de_text]
test_array = []
for item in text_texts_array:
    text = item
    text = re.sub(r'<[^<]+?>', '', text)  # suitable for xml (not applicable here)
    text = text.replace('\n', ' ')  # new lines
    text = re.sub(r'\s+', ' ', text)  # handle multiple spaces
    clean_text = text
    if text_sample_size > len(clean_text) :
        clean_text = ((text_sample_size // len(clean_text)) + 1)*clean_text #repeat
        input_row = get_input_row(clean_text,0,text_sample_size)
        test_array.append(input_row)

test_array = standard_scaler.transform(test_array)
Y_pred = model.predict_classes(test_array)
for id in range(len(test_array)):
    print ("Text:",text_texts_array[id][:50],"... -> Predicted lang: ", decode_langid(Y_pred[id]))