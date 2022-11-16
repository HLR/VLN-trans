from pip import main
from utils import load_datasets, load_nav_graphs
import networkx as nx
import numpy as np
import os
import json
from tqdm import tqdm


    
def cal_dtw(shortest_distances, prediction, reference, threshold=3.0):
    dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
    dtw_matrix[0][0] = 0
    for i in range(1, len(prediction)+1):
        for j in range(1, len(reference)+1):
            best_previous_cost = min(
                dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
            cost = shortest_distances[prediction[i-1]][reference[j-1]]
            dtw_matrix[i][j] = cost + best_previous_cost

    dtw = dtw_matrix[len(prediction)][len(reference)]
    ndtw = np.exp(-dtw/(threshold * len(reference)))
    success = float(shortest_distances[prediction[-1]][reference[-1]] < threshold)
    sdtw = success * ndtw

    return sdtw
    # {
    #     'DTW': dtw,
    #     'nDTW': ndtw,
    #     'SDTW': sdtw
    # }

def cal_cls(shortest_distances, prediction, reference, threshold=3.0):
    def length(nodes):
      return np.sum([
          shortest_distances[a][b]
          for a, b in zip(nodes[:-1], nodes[1:])
      ])

    coverage = np.mean([
        np.exp(-np.min([  # pylint: disable=g-complex-comprehension
            shortest_distances[u][v] for v in prediction
        ]) / threshold) for u in reference
    ])
    expected = coverage * length(reference)
    score = expected / (expected + np.abs(expected - length(prediction)))
    return coverage * score


if __name__ == '__main__':
    ground_path = "/localscratch/zhan1624/VLN-speaker/data/r4r/R4R_val_seen.json"
    predict_path = "/localscratch/zhan1624/VLN-speaker/snap/r4r_test/submit_val_seen.json"
    fake_path = "/localscratch/zhan1624/VLN-speaker/snap/hamt/R4R_val_unseen_sampled_enc.json"
    
    all_scenery_list = os.listdir("/egr/research-hlr/joslin/Matterdata/v1/scans/")
    scans = [i for i in all_scenery_list if i!=".vscode"]
    graphs = load_nav_graphs(scans)
    distances = {}
    positions = {}
    for scan,G in graphs.items(): # compute all shortest paths
        distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
        
    with open(ground_path) as f1, open(predict_path) as f2, open(fake_path) as f3:
        unseen_ground = json.load(f1)
        prediction_data = json.load(f2)
        fake_data = json.load(f3)
    fake_list = []
    for item in fake_data:
        fake_list.append(str(item['path_id']))
        
    ground_dict = {}
    scan_dict = {}
    total_cls = 0
    total_ndtw = 0
    for item in unseen_ground:
        ground_dict[str(item['path_id'])]= item['path']
        scan_dict[str(item['path_id'])] = item['scan']

    count = 0
    for pred in tqdm(prediction_data):
        id = pred['instr_id'].split("_")[0]
        input_scan = scan_dict[id]
        item_pred = list(list(zip(*pred['trajectory']))[0])
        total_cls += cal_cls(distances[input_scan], item_pred, ground_dict[id])
        total_ndtw += cal_dtw(distances[input_scan], item_pred, ground_dict[id] )
        count += 1

    print(total_cls/count)
    print(total_ndtw/count)
    print('yue')








            
