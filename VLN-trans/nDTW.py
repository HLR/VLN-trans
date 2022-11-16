from utils import load_datasets, load_nav_graphs
import networkx as nx
import numpy as np
import os
import json


all_scenery_list = os.listdir("/egr/research-hlr/joslin/Matterdata/v1/scans/pre-trained2/")
scans = [i[:-4] for i in all_scenery_list]
graphs = load_nav_graphs(scans)
distances = {}
positions = {}
for scan,G in graphs.items(): # compute all shortest paths
    distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
    

def ndtw(scan, prediction, reference):
    dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
    dtw_matrix[0][0] = 0
    for i in range(1, len(prediction) + 1):
        for j in range(1, len(reference) + 1):
            best_previous_cost = min(dtw_matrix[i - 1][j],
                                    dtw_matrix[i][j - 1],
                                    dtw_matrix[i - 1][j - 1])
            cost = distances[scan][prediction[i - 1]][reference[j - 1]]
            dtw_matrix[i][j] = cost + best_previous_cost
    dtw = dtw_matrix[len(prediction)][len(reference)]
    ndtw = np.exp(-dtw / (3 * len(reference)))
    return ndtw

def get_position(scan, viewpoints):
    position_list = []
    for each_viewpoint in viewpoints:
        position_list.append(graphs[scan].nodes[each_viewpoint]['position'][:2])
    return position_list

result = 0
sdtw = 0
with open('/VL/space/zhan1624/R2R-EnvDrop/tasks/R2R/data/R2R_val_unseen.json') as f1:
    unseen_ground = json.load(f1)
# 548 only object related direction

with open("/VL/space/zhan1624/obj-vln/snap/agent/result/20210729-193548/submit_val_unseen.json") as f2:
    prediction_data = json.load(f2)





for each_data in prediction_data:
    prediction = []
    for id, each_element in enumerate(each_data["trajectory"]):
        if id == 0:
            prediction.append(each_element[0])
        else:
            if each_element[0] != prediction[-1]:
                prediction.append(each_element[0])
    for each_ground in unseen_ground:
        if each_data['instr_id'].split('_')[0] == str(each_ground["path_id"]):
            tmp_ndtw =  ndtw(each_ground['scan'], prediction, each_ground['path'])
            goal = each_ground['path'][-1]
            final_position = prediction[-1]
            nav_error = distances[each_ground['scan']][final_position][goal]
            sr = nav_error < 3
            sdtw += tmp_ndtw * sr
            result += tmp_ndtw
            break


print(sdtw/len(prediction_data))
print(result/len(prediction_data))
print('yue')
            
