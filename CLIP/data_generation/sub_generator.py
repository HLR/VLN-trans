from ast import dump
import numpy as np
import json
from tqdm import tqdm
import copy

cand_path = "/egr/research-hlr/joslin/candidate.npy"
cand_data = np.load(cand_path, allow_pickle=True).item()

data_path = "/localscratch/zhan1624/CLIP/data_generation/raw_data/R2R_fake_test.json"
with open(data_path) as f_in:
    data = json.load(f_in)
print("---Finish Loading Data---")

landmark_path = "/localscratch/zhan1624/CLIP/data_generation/outputs/all_output.json"
with open(landmark_path) as f_l:
    landmarks = json.load(f_l)
print("---Finish Loading Landmarks---")

motion_path = "/localscratch/zhan1624/CLIP/data_generation/dictionary/motion_verbs.json"
with open(motion_path) as f_m:
    motions = json.load(f_m)

def head_to_index(radius):
    import math
    degree = math.degrees(radius)
    start, end = 12, 24
    for i in range(start, end):
        base = i-12
        if degree >= 30*base and degree <= 30*(base+1):
            return i

def generation(motion_list, landmark_list):
    import random
    random.seed(5)
    motion = random.choice(motion_list)
    landmark = random.choice(landmark_list)
    return motion, landmark

    

if __name__ == '__main__': 
    output_data = []
    for item in tqdm(data):
        new_data = item.copy()
        tmp_motion = copy.deepcopy(motions)
        scan = item['scan']
        path = item['path']
        heading = head_to_index(item['heading'])
        tmp_instr = []
        for view_id, view in enumerate(path):
            candidates = cand_data[scan][view]
            if view_id == len(path)-1:
                landmark_list = landmarks[scan+"_"+view+"_"+str(heading)]
                direct_key = "STOP"
            else:
                for each_can in candidates:
                    if each_can['viewpointId'] == path[view_id+1]:
                        landmark_list = landmarks[scan+"_"+view+"_"+str(each_can['pointId'])]
                        next_heading = each_can['pointId']
                        if next_heading - heading > 3 and next_heading - heading < 11:
                            direct_key = "RIGHT"
                        elif  next_heading - heading < -3 and next_heading - heading > -11:
                            direct_key = "LEFT"
                        elif next_heading - heading < -11:
                            direct_key = "DOWN"
                        elif next_heading - heading > 11:
                            direct_key = "UP"
                        else:
                            direct_key = "FORWARD"
            heading = next_heading
            if len(tmp_motion[direct_key]) == 0:
                print('yue')
            motion, landmark = generation(tmp_motion[direct_key], landmark_list)
            tmp_motion[direct_key].remove(motion)
            tmp_instr.append(motion+" "+landmark)
        new_data['generate_instr'] = tmp_instr
        output_data.append(new_data)
    with open("/localscratch/zhan1624/CLIP/data_generation/new_data/R2R_fake_test_new.json", 'w') as f_out:
        json.dump(output_data, f_out, indent=4)
        

              
