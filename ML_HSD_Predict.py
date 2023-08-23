import pickle
import os
import logging
import configparser
import ML_data_encoders as DataEncoders
import pandas as pd
import numpy as np
import sklearn
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import metrics
import datetime
import requests
from requests_kerberos import HTTPKerberosAuth
import urllib3

# Create a new ConfigParser instance
config = configparser.ConfigParser()
config.read('config.ini')

# Access data from the INI file
workdir = config.get('Workspace', 'directory')
log_filename = os.path.join(workdir, config.get('Workspace', 'execution_log_filename'))
stored_model = os.path.join(workdir, config.get('Model', 'stored_model_file'))
stored_encoder_component = os.path.join(workdir, config.get('Model', 'stored_encoder_component_file'))
stored_encoder_hardware = os.path.join(workdir, config.get('Model', 'stored_encoder_hardware_file'))
stored_encoder_team = os.path.join(workdir, config.get('Model', 'stored_encoder_team_file'))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # To print logs to stdout
        logging.FileHandler(log_filename)  # To log to a file in the current directory
    ]
)

# this is to ignore the ssl insecure warning as we are passing in 'verify=false'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

headers = {'Content-type': 'application/json'}

logging.info(f"Will use ML_data_encoders in version :{DataEncoders.__version__}")
logging.info(f"sklearn version:', {sklearn.__version__}")

# Loading Encoders and Model
model = tf.keras.models.load_model(stored_model)

with open(stored_encoder_component, 'rb') as pk_file:
    encoder_component = pickle.load(pk_file)
with open(stored_encoder_hardware, 'rb') as pk_file:
    encoder_hardware = pickle.load(pk_file)
with open(stored_encoder_team, 'rb') as pk_file:
    encoder_team = pickle.load(pk_file)

logging.info(f"Model loaded")

print(model.summary())


def update_hsd(hsd, val_domain, update_tag=True):
    hsd_id = hsd['id']
    tag_list = hsd['tag'].split(',')
    hsd_url = 'https://hsdes-api.intel.com/rest/article/' + hsd_id
    logging.info(f"attempt to update {hsd_url} , "
                 f"update val_team_owner to {val_domain} ")

    if 'ValTeamAutoAssign' in tag_list:
        raise Exception("Article already tagged, You should not be here")
    tag_list.append('ValTeamAutoAssign')
    tag_list = ",".join(tag_list)
    print('    new tags:', tag_list)

    # create payload
    if update_tag:
        payload = '''
        {
          "tenant": "server_platf_lan",
          "subject": "bug",
          "fieldValues": [
            {
              "server_platf_lan.bug.val_team_owner": "''' + val_domain + '''"
            },
            {
              "tag": "''' + tag_list + '''"
            },
            {
              "send_mail": "false"
            }
    
          ]
        }
       '''
    else:
        payload = '''
         {
           "tenant": "server_platf_lan",
           "subject": "bug",
           "fieldValues": [
             {
               "server_platf_lan.bug.val_team_owner": "''' + val_domain + '''"
             },          
             {
               "send_mail": "false"
             }
           ]
         }
        '''

    update_response = requests.put(hsd_url, verify=False, auth=HTTPKerberosAuth(), headers=headers, data=payload)
    if update_response.status_code == 200:
        logging.info(F"HSD-ES record:{hsd_id} updated - server response.status_code: {update_response.status_code}")
        print("---------------------------------")
        print("|          ", hsd_id, " UPDATED|")
        print("---------------------------------")

    else:
        print(update_response.text)
        logging.error(F"Cannot update record:{hsd_id} - server response.status_code: {update_response.status_code}")
        # raise "Cannot update article "


def ml_predict(hsd):
    """
Returns prediction array
    :param hsd:
    """
    # Casting into data_frame
    print('URL           :', ('https://hsdes.intel.com/appstore/article/#/' + hsd['id']))
    print('title         :', hsd['title'])
    print('component     :', hsd['component'])
    print('team_found    :', hsd['team_found'])
    print('hardware      :', hsd['hardware'])
    print('target_project:', hsd['target_project'])
    # print('val_team_owner:', hsd['val_team_owner'])
    # print('TAGs          :', hsd['tag'])

    # workaround for requirements
    if hsd['team_found'] == 'N/A': hsd['team_found'] = 'SW_Val'
    if hsd['hardware'] == '': hsd['hardware'] = 'No Specific Hardware'
    if hsd['operating_system'] == '': hsd['os_affected']

    data_frame = pd.DataFrame.from_dict({
        'target_project': [hsd['target_project']],
        'component': [hsd['component']],
        'hardware': [hsd['hardware']],
        'team_found': [hsd['team_found']],
        'operating_system': [hsd['operating_system']],
        'title': [hsd['title']],
        'description': [hsd['description']]
    })
    try:
        encoded_component = encoder_component.transform(data_frame[['component', 'team_found']])
    except:
        log_message = f"[component] or [team_found] not encounter before.. need to retrain model\n" \
                      f"    component       : {hsd['component']} \n" \
                      f"    team_found      :'{hsd['team_found']}\n"
        print('------------------ Exception ------------------------')
        print(log_message)
        logging.error(log_message)
        print('-----------------------------------------------------')
        return [(0, 'Error - Missing component or team_found')]
    try:
        encoded_hardware = encoder_hardware.transform(data_frame[['hardware', 'target_project']])
    except:
        log_message = f"[hardware] or [target_project] not encounter before.. need to retrain model \n" \
                      f"    hardware       : {hsd['hardware']} \n" \
                      f"    target_project :'{hsd['target_project']}\n"
        print('------------------ Exception ------------------------')
        print(log_message)
        logging.error(log_message)
        print('-----------------------------------------------------')
        return [(0, 'Error - Missing hardware or target_project')]

    encoded_title = DataEncoders.encoder_keyword(data_frame['title'], DataEncoders.dictionary)
    encoded_description = DataEncoders.encoder_keyword(data_frame['description'], DataEncoders.full_dictionary)
    encoded_os = DataEncoders.encoder_mapping(data_frame['operating_system'], DataEncoders.os_mapping, "OS mapping")

    # Make predictions
    predictions = model.predict(
        {"component": encoded_component,
         'hardware': encoded_hardware,
         'title': encoded_title,
         "os": encoded_os,
         'description': encoded_description})

    # Get the indices of the top N predictions
    top_n = 5
    top_indices = np.argsort(predictions[0])[::-1][:top_n]

    # Map indices to category names
    categories = encoder_team.categories_[0]

    return [(predictions[0][i] * 100, categories[i]) for i in top_indices]


current_datetime = datetime.datetime.now()
week_number = current_datetime.strftime("%U")
log_message = f"execution started WW{week_number} [{current_datetime.strftime('%m/%d/%Y')}]" \
              f" - {current_datetime.strftime('%H:%M')}"

logging.info(f"----------------------------------------------------------------")
logging.info(f"----------------------------------------------------------------")
logging.info(log_message)
logging.info(f"----------------------------------------------------------------")
logging.info(f"----------------------------------------------------------------")

# and Load model and encoders

# New and updated sw defects 24h
# https://hsdes.intel.com/appstore/community/#/1208188470?queryId=18013029829
query_ID = '18013029829'  # New and updated sw defects 24h
# query_ID = '18032780448'  # requirements

url = 'https://hsdes-api.intel.com/rest/query/execution/' + query_ID
print('attempt to get query:', url)
print('https://hsdes.intel.com/appstore/community/#/1208188470?queryId=18013029829')

response = requests.get(url, verify=False, auth=HTTPKerberosAuth(), headers=headers)
print("response status code:", response.status_code)
if 200 == response.status_code:
    data_rows = response.json()['data']
    print('retrieved ', len(data_rows), ' articles')
else:
    raise "Cannot execute query "

print('retrieved ', len(data_rows), ' articles')
for i, article in enumerate(data_rows):
    print("")
    print(i + 1, '/', len(data_rows), ' [', article['val_team_owner'], ']', article['id'], article['title'][:72])
    logging.info(f"[{i+1}/{len(data_rows)}] - {article['id']} {article['title'][:72]} .")
    top_prediction = ml_predict(article)

    top_probability = top_prediction[0][0]  # Probability value of the top prediction
    if top_probability < 10:
        logging.warning("Skipping HSD-UPDATE")
        input(f"Press ENTER to continue....")
        continue

    for position, prediction in enumerate(top_prediction):
        print(f"{position + 1}. Prediction: {prediction[0]:.2f}% | Category: {prediction[1]}")

    if top_probability > 90:
        logging.info('probability > 90% - updating without asking')
        update_hsd(article, top_prediction[0][1], True)  # adding TAG to exclude article from training data.
    else:
        logging.info('Supervision needed')
        print('0. No SW Validation required')
        print('N. Skip record')
        choice = input(f"Choose a team - default {top_prediction[0][1]}")
        if choice == '':
            choice = "1"
        if choice == "N":
            continue
        elif choice in ["1", "2", "3", "4", "5"]:
            validationDomain = top_prediction[int(choice) - 1][1]
            update_hsd(article, validationDomain, False)
        elif choice == "0":
            update_hsd(article, 'No Validation required', True)
        else:
            print("Invalid choice")
