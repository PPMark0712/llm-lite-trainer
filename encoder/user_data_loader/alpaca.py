# alpaca
import json


def data_loader(fn):
    with open(fn, "r") as f:
        data = json.load(f)
    for item in data:
        yield {
            "q": item["instruction"],
            "a": item["input"] + item["output"]
        }


def get_file_list():
    file_list = ["/data1/yyz/downloads/datasets/yahma/alpaca-cleaned/alpaca_data_cleaned.json"]
    return file_list