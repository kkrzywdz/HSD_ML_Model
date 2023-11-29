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
req_df = reqs[['id', 'title', 'description']].copy()
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
output = out_team_encoded.shape[1] * 8
lstm_units = 300
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
embedded_title = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=output*2)(input_title_token)
embedded_description = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=output*2)(input_description_token)


lstm_title = LSTM(units=lstm_units)(embedded_title)
lstm_description = LSTM(units=lstm_units)(embedded_description)
concatenated_title_desc_token = Concatenate()([lstm_title, lstm_description])

dense_layer_title_desc_token = tf.keras.layers.Dense(2 * output, activation='relu')(concatenated_title_desc_token)

dense_layer_title = tf.keras.layers.Dense(2 * output, activation='relu')(input_title)
dense_layer_description = tf.keras.layers.Dense(2 * output, activation='relu')(input_description)

concatenated_title_desc = tf.keras.layers.concatenate([dense_layer_title, dense_layer_description])
dense_layer_title_desc = tf.keras.layers.Dense(2 * output, activation='relu')(concatenated_title_desc)

concatenated_all = tf.keras.layers.concatenate([dense_layer_title_desc, dense_layer_title_desc_token])

# Additional layers on the concatenated output
dense_layer_all1 = tf.keras.layers.Dense(4 * output, activation='relu')(concatenated_all)
dense_layer_all2 = tf.keras.layers.Dense(3 * output, activation='relu')(dense_layer_all1)


# Output layer
output = tf.keras.layers.Dense(out_team_encoded.shape[1], activation='softmax',
                               name='val_domain')(dense_layer_all2)

#output = tf.keras.layers.Dense(out_team_encoded.shape[1], activation=None,
#                               name='val_domain')(dense_layer_all2)

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


in_title_encoded = DataEncoders.encoder_keyword(req_df['title'], DataEncoders.dictionary)
in_description_encoded = DataEncoders.encoder_keyword(req_df['description'], DataEncoders.full_dictionary)

predictions = model.predict(
    {
     'title': in_title_encoded,
     'title_token': req_df['title_copy'],
     'description': in_description_encoded,
     'description_token': req_df['des_copy']},
    )

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'ID': req_df['id'],
    'Title': req_df['title'],
})

# Add columns for each output category
for i in range(37):  # Assuming 37 output categories
    cat = encoder_team.categories_[0][i]
    results_df[f'{cat} - {i+1}'] = predictions[:, i]

# Save the results to a CSV file
results_df.to_csv('predicted_results.csv', index=False)



'''# Make predictions
    predictions = model.predict(
        {"component": encoded_component,
         "team_found": encoded_team_found,
         'hardware': encoded_hardware,
         "target_project": encoded_target_project,
         'title': encoded_title,
         'os': encoded_os,
         'description': encoded_description})

    # Get the indices of the top N predictions
    top_n = 5
    top_indices = np.argsort(predictions[0])[::-1][:top_n]

    # Map indices to category names
    categories = encoder_team.categories_[0]

    return [(predictions[0][i] * 100, categories[i]) for i in top_indices]'''


# Save trained model anf fitted encoders

