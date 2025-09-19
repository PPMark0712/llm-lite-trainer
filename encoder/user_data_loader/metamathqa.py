# metamathqa
import json


def data_loader(fn):
    fn = "/data1/yyz/downloads/datasets/MetaMathQA/MetaMathQA-395K.json"
    with open(fn, "r") as f:
        data = json.load(f)
    for item in data:
        yield {
            "q": item["query"],
            "a": item["response"]
        }
