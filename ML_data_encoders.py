# common defect encoders

import logging
import configparser
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import re


# pattern = r'\d+(\.\d+)+'
# Projects = data['target_project'].apply(lambda x: re.sub(pattern, '', x)).unique()
# np.sort(Projects)

def remove_versions(project_name):
    return re.sub(r'\s\d+(\.\d+)+$', '', project_name)

class HSDEncoders:
    def __init__(self):

        self.stored_encoder_target_project = None
        self.stored_encoder_hardware = None
        self.stored_encoder_team_found = None
        self.stored_encoder_component = None
        self.stored_encoder_team = None

        self.encoder_component = None
        self.encoder_team_found = None
        self.encoder_hardware = None
        self.encoder_target_project = None
        self.encoder_os = None
        self.encoder_title = None
        self.encoder_description = None

        self.encoder_team = None

    def parse_config_file(self,ini_file ):
        print(f"Attempt to load Encoders from file: {ini_file}")
        config = configparser.ConfigParser()
        config.read(ini_file)
        workdir = config.get('Workspace', 'directory')
        self.stored_encoder_component = os.path.join(workdir, config.get('Model', 'stored_encoder_component_file'))
        self.stored_encoder_team_found = os.path.join(workdir, config.get('Model', 'stored_encoder_team_found_file'))
        self.stored_encoder_hardware = os.path.join(workdir, config.get('Model', 'stored_encoder_hardware_file'))
        self.stored_encoder_target_project = os.path.join(workdir,
                                                          config.get('Model', 'stored_encoder_target_project_file'))
        self.stored_encoder_team = os.path.join(workdir, config.get('Model', 'stored_encoder_team_file'))

    def load_encoders (self):
        with open(self.stored_encoder_component, 'rb') as pk_file:
            self.encoder_component = pickle.load(pk_file)
        with open(self.stored_encoder_team_found, 'rb') as pk_file:
            self.encoder_team_found = pickle.load(pk_file)
        with open(self.stored_encoder_hardware, 'rb') as pk_file:
            self.encoder_hardware = pickle.load(pk_file)
        with open(self.stored_encoder_target_project, 'rb') as pk_file:
            self.encoder_target_project = pickle.load(pk_file)
        with open(self.stored_encoder_team, 'rb') as pk_file:
            self.encoder_team = pickle.load(pk_file)

    def save_encoders(self):
        with open(self.stored_encoder_component, 'wb') as f:
            pickle.dump(self.encoder_component, f)
        with open(self.stored_encoder_team_found, 'wb') as f:
            pickle.dump(self.encoder_team_found, f)
        with open(self.stored_encoder_hardware, 'wb') as f:
            pickle.dump(self.encoder_hardware, f)
        with open(self.stored_encoder_target_project, 'wb') as f:
            pickle.dump(self.encoder_target_project, f)
        with open(self.stored_encoder_team, 'wb') as f:
            pickle.dump(self.encoder_team, f)

    def initialize_encoders(self):
        self.encoder_component = OneHotEncoder(sparse_output=False)
        self.encoder_team_found = OneHotEncoder(sparse_output=False)
        self.encoder_hardware = OneHotEncoder(sparse_output=False)
        self.encoder_target_project = OneHotEncoder(sparse_output=False)
        self.encoder_team = OneHotEncoder(sparse_output=False)

    def fit_encoder_component(self, input_data):
        self.encoder_component.fit(input_data)

    def fit_encoder_team_found(self, input_data):
        self.encoder_team_found.fit(input_data)

    def fit_encoder_hardware(self, input_data):
        self.encoder_hardware.fit(input_data)

    def fit_encoder_target_project(self, input_data):
        pattern = r'\d+(\.\d+)+'
        input_data = input_data['target_project'].apply(lambda x: re.sub(pattern, '', x)).unique()
        self.encoder_target_project.fit(input_data)

    def fit_encoder_team(self, input_data):
        self.encoder_team_found.fit(input_data)

import ML_data_dictionary as data_dictionary
import ML_data_mapping as data_mapping

__version__ = 0.8

print('ML_data_dictionary', data_dictionary.__version__)
print('ML_data_mapping', data_mapping.__version__)
print('ML_data_encoders', __version__)

dictionary = data_dictionary.tag_list + data_dictionary.other_list

# to remove duplicates  (will not use tags in description)
full_dictionary = list(set(data_dictionary.other_list + data_dictionary.full_list))

os_mapping = data_mapping.os_mapping


# keyword encoder
def encoder_keyword(df_text, keywords):
    # Initialize an empty list to store the binary arrays for each string
    binary_arrays = []

    # Iterate through each string in the array and find the keywords
    for line in df_text.values:
        binary_array = [int(keyword in line.lower()) for keyword in keywords]
        binary_arrays.append(binary_array)

    # Convert the list of binary arrays to a 2D NumPy array
    return np.array(binary_arrays)


# mapping encoder
def mach_of(single_line, keys, mapping):
    for key in (mapping[keys]):
        if key in single_line:
            return 1
    return 0


def encoder_mapping(df_data, mapping, source='OS Mapping'):
    # Initialize an empty list to store the binary arrays for each string
    binary_arrays = []

    # Iterate through each string in the array and find the keywords
    for line in df_data.values:
        binary_array = [int(mach_of(line.lower(), broad_category, mapping)) for broad_category in mapping]
        binary_arrays.append(binary_array)
        if np.sum(binary_array) == 0:
            log_message = f"Missing item '{line}' in the mapping from {source}."
            logging.warning(log_message)

    # Convert the list of binary arrays to a 2D NumPy array
    return np.array(binary_arrays)
