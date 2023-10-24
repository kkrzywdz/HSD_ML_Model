import pickle
import re
import os
import logging
import configparser
import ML_data_encoders as DataEncoders
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Concatenate
from tensorflow.keras.models import Model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import metrics
import datetime

# NO CONFIG
# Remove Logging

print(f"TensorFlow version:{tf.__version__}")
print(f"pandas version:{pd.__version__}")
print(f"numpy version:{ np.__version__}")
print(f"defect encoder version{ DataEncoders.__version__}")

training_data_file = 'c:\Sandbox\Data\ML Training data WW43.csv'
req_data_file ='c:\Sandbox\Data\Validation MGV PRDs.csv'

# Load the data from CSV file
print(f"loading data from :{training_data_file}")
data = pd.read_csv(training_data_file)
reqs = pd.read_csv(req_data_file)

# print (data)

# Split the data into input features (input) and target variable (output)
input_df = data[['title', 'description']].copy()
req_df = reqs[['title', 'description']].copy()
output_df = data[['val_team_owner']].copy()

input_df['des_copy'] = data['description'].apply(lambda x: re.sub(r'[^A-Za-z\s]', ' ', x)).str.replace('\xa0', ' ')
input_df['title_copy'] = data['title'].apply(lambda x: re.sub(r'[^A-Za-z\s]', ' ', x)).str.replace('\xa0', ' ')
req_df['des_copy'] = req_df['description'].apply(lambda x: re.sub(r'[^A-Za-z\s]', ' ', x)).str.replace('\xa0', ' ')
req_df['title_copy'] = req_df['title'].apply(lambda x: re.sub(r'[^A-Za-z\s]', ' ', x)).str.replace('\xa0', ' ')

tokenizer = Tokenizer()
all_data = pd.concat([input_df, req_df], axis=0, ignore_index=True)

tokenizer.fit_on_texts(all_data['des_copy'] + all_data['title_copy'])

# Access the vocabulary
word_index = tokenizer.word_index
print("word index got",len(word_index), 'words')


input_df['des_copy'] = tokenizer.texts_to_sequences(input_df['des_copy'])
req_df['des_copy'] = tokenizer.texts_to_sequences(req_df['des_copy'])
input_df['title_copy'] = tokenizer.texts_to_sequences(input_df['title_copy'])
req_df['title_copy'] = tokenizer.texts_to_sequences(req_df['title_copy'])

max_length_title = 50
input_df['title_copy'] = pad_sequences(input_df['title_copy'], maxlen=max_length_title, padding='post')
req_df['title_copy'] = pad_sequences(req_df['title_copy'], maxlen=max_length_title, padding='post')

max_length_desc = 100
input_df['des_copy'] = pad_sequences(input_df['des_copy'], maxlen=max_length_desc, padding='post')
req_df['des_copy'] = pad_sequences(req_df['des_copy'], maxlen=max_length_desc, padding='post')

# Apply the function to the 'project' column
encoder_team = OneHotEncoder(sparse_output=False)
encoder_team.fit(output_df[['val_team_owner']])

# Split the data into train and test sets
logging.info(f"Split the data into train and test sets")
in_train, in_test, out_train, out_test = train_test_split(input_df, output_df, test_size=0.10, random_state=32)

print('in_train shape :', in_train.shape)
print('in_test shape  :', in_test.shape)
print('out_train shape:', out_train.shape)
print('out_test shape :', out_test.shape)

# Encode all data..
# train data Encoding

in_title_encoded = DataEncoders.encoder_keyword(in_train['title'], DataEncoders.dictionary)
in_description_encoded = DataEncoders.encoder_keyword(in_train['description'], DataEncoders.full_dictionary)
out_team_encoded = encoder_team.transform(out_train)

test_in_title_encoded = DataEncoders.encoder_keyword(in_test['title'], DataEncoders.dictionary)
test_in_description_encoded = DataEncoders.encoder_keyword(in_test['description'], DataEncoders.full_dictionary)

test_out_team_encoded = encoder_team.transform(out_test)
output = out_team_encoded.shape[1] * 4
lstm_units = 100
vocab_size = len(word_index)

print(f"\n\nbuilding the model\n\n")
# defining inputs
print('Defining inputs')

#inputs

input_title = tf.keras.Input(shape=in_title_encoded.shape[1], name='title')
input_description = tf.keras.Input(shape=in_description_encoded.shape[1], name='description')

input_title_token = tf.keras.Input(shape=(max_length_title,), name='title_token')
input_description_token = tf.keras.Input(shape=(max_length_desc,), name='description_token')

# Embedding layer for text data
embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=output*4)

embedded_title = embedding_layer(input_title)
embedded_description = embedding_layer(input_description)

# LSTM layers for text data
lstm_units = 100  # You can adjust this value as needed

lstm_title = LSTM(units=lstm_units)(embedded_title)
lstm_description = LSTM(units=lstm_units)(embedded_description)
concatenated_title_desc_token = Concatenate()([lstm_title, lstm_description])

dense_layer_title_desc_token = tf.keras.layers.Dense(2 * output, activation='relu')(concatenated_title_desc_token)

dense_layer_title = tf.keras.layers.Dense(4 * output, activation='relu')(input_title)
dense_layer_description = tf.keras.layers.Dense(4 * output, activation='relu')(input_description)

concatenated_title_desc = tf.keras.layers.concatenate([dense_layer_title, dense_layer_description])
dense_layer_title_desc = tf.keras.layers.Dense(2 * output, activation='relu')(concatenated_title_desc)

concatenated_all = tf.keras.layers.concatenate([dense_layer_title_desc, dense_layer_title_desc_token])

# Additional layers on the concatenated output
dense_layer_all1 = tf.keras.layers.Dense(4 * output, activation='elu')(concatenated_all)
dense_layer_all2 = tf.keras.layers.Dense(2 * output, activation='elu')(dense_layer_all1)

# Output layer
output = tf.keras.layers.Dense(out_team_encoded.shape[1], activation='softmax',
                               name='val_domain')(dense_layer_all2)

# Create the model
model = tf.keras.Model(
    inputs=[input_title,
            input_title_token,
            input_description,
            input_description_token],
    outputs=output
)

# Compile the model
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy', metrics.Precision()])

# Evaluate the model
model.summary()

tf.keras.utils.plot_model(model, 'reqmodel.jpg', show_shapes=True)


logging.info(f"Training the model...")

# Train the model
model.fit(
    {
     'title': in_title_encoded,
     'title_token': in_train['title_copy'],
     'description': in_description_encoded,
     'description_token': in_train['des_copy']},
    {"val_domain": out_team_encoded},
    epochs=10,
    batch_size=16
)

logging.info(f"Evaluating the model...")

loss, accuracy, precision = model.evaluate(
    {
     'title': test_in_title_encoded,
     'title_token': in_test['title_copy'],
     'description': test_in_description_encoded,
     'description_token': in_test['des_copy']},
    {"val_domain": test_out_team_encoded})
print(f'Test loss: {loss}')
print(f'Test accuracy:{accuracy}')
print(f'Test precision:{precision}')




# Save trained model anf fitted encoders

