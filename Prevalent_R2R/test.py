

from vilmodel import BertImgModel,BertLayerNorm
from pytorch_transformers import BertConfig
import inspect
import pdb


"""
b = BertConfig.from_pretrained('bert-base-uncased')
b.img_feature_dim = 2176
b.img_feature_type = ""

#model1 = BertModel(b)

model = BertImgModel(b)
pdb.set_trace()
"""
# print("you got it")

import json

data = []
with open("/VL/space/zhan1624/PREVALENT_R2R/tasks/R2R/data/collect_traj/shortest_1.json") as f_in:
    data = json.load(f_in)
print('yue')
