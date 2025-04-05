import pandas as pd
import json
df = pd.read_csv("test_set.csv")
df.drop(columns=["nl type"],inplace=True)
df.drop(columns=["table name"], inplace=True)
df.drop(columns=["database"], inplace=True)



output_path = "test_set.json"


# Create a multiline json
record_dict = json.loads(df.to_json(orient = "records"))
record_json = json.dumps(record_dict)

with open(output_path, 'w') as f:
    f.write(record_json)

# def load_json(json_file):
#                     with open(json_file, "r", encoding="utf-8") as fh:
#                         return json.load(fh)


# data_list = load_json('train_set.json')
# docs = [item for item in data_list] 
# print(docs[0])