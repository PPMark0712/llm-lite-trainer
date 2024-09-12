# metamathqa
import json


def data_loader(fn):
    with open(fn, "r") as f:
        data = json.load(f)
    for item in data:
        yield {
            "q": item["query"],
            "a": item["response"]
        }
    

def get_file_list():
    file_list = ["/data1/yyz/downloads/datasets/MetaMathQA/MetaMathQA-395K.json"]
    return file_list

