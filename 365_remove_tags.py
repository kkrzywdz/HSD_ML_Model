
import requests
from requests_kerberos import HTTPKerberosAuth
import urllib3

# this is to ignore the ssl insecure warning as we are passing in 'verify=false'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

headers = {'Content-type': 'application/json'}


def update_hsd(hsd ):
    hsd_id = hsd['id']
    tag_list = hsd['tag'].split(',')
    hsd_url = 'https://hsdes-api.intel.com/rest/article/' + hsd_id
    #print(f"attempt to update {hsd_url} , ")
    try:
        #tag_list.remove('365KeepOpen')
        tag_list.remove('[365KeepOpen]')

    except:
        print(f"Cannot remove tag")
        return 0

    tag_list = ",".join(tag_list)
    print(f"    new tags: {tag_list} ")

    # create payload
    payload = '''
        {
          "tenant": "server_platf_lan",
          "subject": "bug",
          "fieldValues": [            
            {
              "tag": "''' + tag_list + '''"
            },
            {
              "send_mail": "false"
            }

          ]
        }
       '''

    try:
        update_response = requests.put(hsd_url, verify=False, auth=HTTPKerberosAuth(), headers=headers, data=payload)
        if update_response.status_code == 200:
        #    logging.info(F"HSD-ES record:{hsd_id} updated - server response.status_code: {update_response.status_code}")
            print("---------------------------------")
            print("|          ", hsd_id, " UPDATED|")
            print("---------------------------------")
        else:
            print(update_response.text)
            raise "cannot update Article"
    except:
        print("Cannot update this record cause characters in tags")
    #    logging.error(F"Cannot update record:{hsd_id} - server response.status_code: {update_response.status_code}")



query_ID = '18032100212'  # 365 keepOpen

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
    print(i + 1, '/', len(data_rows), article['id'], article['title'][:72])
    print('URL           :', ('https://hsdes.intel.com/appstore/article/#/' + article['id']))
    print(f"    old tags: {article['tag']} ")
    update_hsd(article)





