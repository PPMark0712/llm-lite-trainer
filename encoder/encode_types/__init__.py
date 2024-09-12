from .encode_pretrain import encode_pretrain, data_func_pretrain
from .encode_qa import encode_qa, data_func_qa

encode_config = {
    "pretrain": {
        "encode_func": encode_pretrain,
        "data_func": data_func_pretrain,
    },
    "qa": {
        "encode_func": encode_qa,
        "data_func": data_func_qa,
    }
}