import pickle
import os
import logging
import configparser
import ML_data_encoders as de
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import metrics
import datetime

# Create a new ConfigParser instance
config = configparser.ConfigParser()

# Read the INI data from the file
config.read('config.ini')

# Access data from the INI file
workdir = config.get('Workspace', 'directory')

log_filename = os.path.join(workdir, config.get('Workspace', 'training_log_filename'))
log_model_map_image = os.path.join(workdir, config.get('Workspace', 'model_graph_image'))

training_data_file = os.path.join(workdir, config.get('data', 'training_data'))

stored_model = os.path.join(workdir, config.get('Model', 'stored_model_file'))
stored_encoder_component = os.path.join(workdir, config.get('Model', 'stored_encoder_component_file'))
stored_encoder_hardware = os.path.join(workdir, config.get('Model', 'stored_encoder_hardware_file'))
stored_encoder_team = os.path.join(workdir, config.get('Model', 'stored_encoder_team_file'))

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # To print logs to stdout
        logging.FileHandler(log_filename)  # To log to a file in the current directory
    ]
)

current_datetime = datetime.datetime.now()
week_number = current_datetime.strftime("%U")

log_message = f"execution started WW{week_number} [{current_datetime.strftime('%m/%d/%Y')}]" \
              f" - {current_datetime.strftime('%H:%M')}"

logging.info(f"----------------------------------------------------------------")
logging.info(f"----------------------------------------------------------------")
logging.info(log_message)
logging.info(f"----------------------------------------------------------------")
logging.info(f"----------------------------------------------------------------")

logging.debug("TensorFlow version:", tf.__version__)
logging.debug("pandas version:", pd.__version__)
logging.debug("numpy version:", np.__version__)
logging.debug("defect encoder version", de.__version__)

# Load the data from CSV file
logging.info(f"loading data from :{training_data_file}")
data = pd.read_csv(training_data_file)

# Split the data into input features (input) and target variable (output)
input_df = data[['title', 'component', 'operating_system', 'target_project', 'hardware', 'tag',
                 'team_found', 'description']].copy()
output_df = data[['val_team_owner']].copy()

logging.info(f"input data shape :{input_df.shape}")

# for onehot Single will be enough, but I for training I will use 2

logging.info(f"fitting encoders: encoder_component (component , team_found)")
logging.info(f"fitting encoders: encoder_hardware (hardware , target_project)")
logging.info(f"fitting encoders: encoder_team (val_team_owner)")

encoder_component = OneHotEncoder(sparse_output=False)
# encoder_team_found = OneHotEncoder(sparse_output=False)
encoder_hardware = OneHotEncoder(sparse_output=False)
# encoder_target_project = OneHotEncoder(sparse_output=False)
encoder_team = OneHotEncoder(sparse_output=False)

encoder_component.fit(input_df[['component', 'team_found']])
encoder_hardware.fit(input_df[['hardware', 'target_project']])
encoder_team.fit(output_df[['val_team_owner']])

# Split the data into train and test sets
logging.info(f"Split the data into train and test sets")
in_train, in_test, out_train, out_test = train_test_split(input_df, output_df, test_size=0.05, random_state=42)
print('in_train shape :', in_train.shape)
print('in_test shape  :', in_test.shape)
print('out_train shape:', out_train.shape)
print('out_test shape :', out_test.shape)

logging.info(f"training data shape :{in_train.shape}")
logging.info(f"test data shape :{in_test.shape}")

# Encode all data..
# train data Encoding

in_component_encoded = encoder_component.transform(in_train[['component', 'team_found']])
in_hardware_encoded = encoder_hardware.transform(in_train[['hardware', 'target_project']])
in_title_encoded = de.encoder_keyword(in_train['title'], de.dictionary)
in_description_encoded = de.encoder_keyword(in_train['description'], de.full_dictionary)
in_os_encoded = de.encoder_mapping(in_train['operating_system'], de.os_mapping, "OS mapping")
out_team_encoded = encoder_team.transform(out_train)

# test data Encoding

test_in_component_encoded = encoder_component.transform(in_test[['component', 'team_found']])
test_in_hardware_encoded = encoder_hardware.transform(in_test[['hardware', 'target_project']])
test_in_title_encoded = de.encoder_keyword(in_test['title'], de.dictionary)
test_in_description_encoded = de.encoder_keyword(in_test['description'], de.full_dictionary)
test_in_os_encoded = de.encoder_mapping(in_test['operating_system'], de.os_mapping, "OS mapping")
test_out_team_encoded = encoder_team.transform(out_test)

print('----------------------------------------------')
print('in_component_encoded shape:', in_component_encoded.shape)
print('in_hardware_encoded shape:', in_hardware_encoded.shape)
print('in_title_encoded shape:', in_title_encoded.shape)
print('in_description_encoded shape:', in_description_encoded.shape)
print('in_os_encoded shape:', in_os_encoded.shape)
print('out_team_encoded shape:', out_team_encoded.shape)

print('----------------------------------------------')
print('test_in_component_encoded shape:', test_in_component_encoded.shape)
print('test_in_hardware_encoded shape:', test_in_hardware_encoded.shape)
print('test_in_title_encoded shape:', test_in_title_encoded.shape)
print('test_in_description_encoded shape:', test_in_description_encoded.shape)
print('test_in_os_encoded shape:', test_in_os_encoded.shape)
print('out_team_encoded shape:', out_team_encoded.shape)
print('----------------------------------------------')

# del model

logging.info(f"building the model")
# defining inputs
print('Defining inputs')
input_component = tf.keras.Input(shape=in_component_encoded.shape[1], name='component')
input_hardware = tf.keras.Input(shape=in_hardware_encoded.shape[1], name='hardware')
input_title = tf.keras.Input(shape=in_title_encoded.shape[1], name='title')
input_description = tf.keras.Input(shape=in_description_encoded.shape[1], name='description')
input_os = tf.keras.Input(shape=in_os_encoded.shape[1], name='os')

# Dense layers for continuous input
print('Dense layers for continuous input')
component_dense_output = tf.keras.layers.Dense(72, activation='relu')(input_component)
hardware_dense_output1 = tf.keras.layers.Dense(72, activation='relu')(input_hardware)
hardware_dense_output2 = tf.keras.layers.Dense(10, activation='relu')(hardware_dense_output1)

title_dense_output = tf.keras.layers.Dense(in_title_encoded.shape[1], activation='relu')(input_title)
# description_dense_output = tf.keras.layers.Dense(in_description_encoded.shape[1],
# activation='relu')(input_description)
description_dense_output = tf.keras.layers.Dense(72, activation='relu')(input_description)
# description_dense_output_2 = tf.keras.layers.Dense(2 * in_description_encoded.shape[1],
#                                                   activation='relu')(description_dense_output_1)
# description_dense_output_3 = tf.keras.layers.Dense(72, activation='relu')(description_dense_output_2)

os_dense_output = tf.keras.layers.Dense(in_title_encoded.shape[1], activation='relu')(input_os)

# Concatenate the outputs of Dense layers
print('Concatenate the outputs of Dense layers')
concatenated_output = tf.keras.layers.concatenate(
    [component_dense_output, hardware_dense_output2, title_dense_output, description_dense_output, os_dense_output])
# [componen_dense_output, hardware_dense_output, title_dense_output,input_os])

# Additional layers on the concatenated output
concatenated_output = tf.keras.layers.Dense(64, activation='relu')(concatenated_output)

# Output layer
output = tf.keras.layers.Dense(out_team_encoded.shape[1], activation='softmax', name='val_domain')(concatenated_output)

# Create the model
model = tf.keras.Model(
    inputs=[input_component, input_hardware, input_title, input_os, input_description],
    outputs=output
)

# Compile the model
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy', metrics.Precision()])

# Evaluate the model
model.summary()

tf.keras.utils.plot_model(model, log_model_map_image, show_shapes=True)
logging.info(f"model image stored in :{log_model_map_image}")

logging.info(f"Training the model...")

# Train the model
model.fit(
    {"component": in_component_encoded,
     'hardware': in_hardware_encoded,
     'title': in_title_encoded,
     'os': in_os_encoded,
     'description': in_description_encoded},
    {"val_domain": out_team_encoded},
    epochs=10,
    batch_size=16
)

logging.info(f"Evaluating the model...")

loss, accuracy, precision = model.evaluate(
    {"component": test_in_component_encoded,
     'hardware': test_in_hardware_encoded,
     'title': test_in_title_encoded,
     "os": test_in_os_encoded,
     'description': test_in_description_encoded},
    {"val_domain": test_out_team_encoded})
logging.info(f'Test loss: {loss}')
logging.info(f'Test accuracy:{accuracy}')
logging.info(f'Test precision:{precision}')

# Save trained model anf fitted encoders

model.save(stored_model)
with open(stored_encoder_component, 'wb') as f:
    pickle.dump(encoder_component, f)
with open(stored_encoder_hardware, 'wb') as f:
    pickle.dump(encoder_hardware, f)
with open(stored_encoder_team, 'wb') as f:
    pickle.dump(encoder_team, f)

logging.info(f'Model stored in : {stored_model}')
logging.info(f'component amd team_found encoder stores in : {stored_encoder_component}')
logging.info(f'hardware and project encoder stores in : {stored_encoder_hardware}')
logging.info(f'val team encoder stores in : {stored_encoder_team}')

#
#
#
# HERE WILL BE MARKUP TO TRY USING MODEL WITH
'''
HSD = {
    'target_project': '10GbE SW Backlog',
    'component': 'sw.40g_ndis_i40ea',
    'hardware': 'Amber Acres',
    'team_found': 'SW',
    'operating_system': 'operating_system',
    'title': '[P4 Compiler] Attempting to do both variable length',
    'description': '[P4 Compiler] Attempting to do both variable length extraction & set '
                   'state variable in same state is broken '
}

# Casting into data_frame

data_frame = pd.DataFrame.from_dict({
    'target_project': [HSD['target_project']],
    'component': [HSD['component']],
    'hardware': [HSD['hardware']],
    'team_found': [HSD['team_found']],
    'operating_system': [HSD['operating_system']],
    'title': [HSD['title']],
    'description': [HSD['description']]
})

print(HSD)
print(data_frame)  # 1 rows x 7 columns
print('==================================================')
print("data_frame['component']:", data_frame)
print('==================================================')

config_file = configparser.ConfigParser()

# Read the INI data from the file
config_file.read('config.ini')

workdir = config_file.get('Workspace', 'directory')
stored_model = os.path.join(workdir, config_file.get('Model',
                                                     'stored_model_file'))
stored_encoder_component = os.path.join(workdir, config_file.get('Model',
                                                                 'stored_encoder_component_file'))
stored_encoder_hardware = os.path.join(workdir, config_file.get('Model',
                                                                'stored_encoder_hardware_file'))
stored_encoder_team = os.path.join(workdir, config_file.get('Model',
                                                            'stored_encoder_team_file'))

# Loading Encoders

new_model = tf.keras.models.load_model(stored_model)

with open(stored_encoder_component, 'rb') as pk_file:
    encoder_component = pickle.load(pk_file)
with open(stored_encoder_hardware, 'rb') as pk_file:
    encoder_hardware = pickle.load(pk_file)
with open(stored_encoder_team, 'rb') as pk_file:
    encoder_team = pickle.load(pk_file)

encoded_component_encoded = encoder_component.transform(data_frame[['component', 'team_found']])
encoded_hardware = encoder_hardware.transform(data_frame[['hardware', 'target_project']])

'''