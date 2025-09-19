# alpaca
import json


def data_loader():
    fn = "/data1/yyz/downloads/datasets/yahma/alpaca-cleaned/alpaca_data_cleaned.json"
    with open(fn, "r") as f:
        data = json.load(f)
    for item in data:
        yield {
            "q": item["instruction"],
            "a": item["input"] + item["output"]
        }
