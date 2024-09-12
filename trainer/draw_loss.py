import json, os
from matplotlib import pyplot as plt
import numpy as np


def draw_loss(ckpt_path):
    loss_fn = os.path.join(ckpt_path, "loss_list.json")
    with open(loss_fn, "r") as f:
        l = json.load(f)
        
    plt.clf()
    
    # 折线图保留n_points个点，否则上万条数据的折线图会很乱
    x, y = [], []
    n_points = min(100, len(l))  # 图上保留几个点
    step = len(l) // n_points 
    for i in range(0, len(l), step):
        x.append(i + 1)
        y.append(np.mean(l[i:i + step]))  # 取一段loss的平均值，而非单点取值

    # 绘制loss图片
    plt.title("train loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.plot(x, y)

    # 将loss图片保存在对应checkpoint文件夹中
    fn = os.path.join(ckpt_path, "loss_img.png")
    # print("save loss picture at:", fn)
    plt.savefig(fn)
   


if __name__ == "__main__":
    # 读取loss_list
    ckpt_path = ""
    draw_loss(ckpt_path)