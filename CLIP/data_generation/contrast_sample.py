from ast import dump
import numpy as np
import json
from tqdm import tqdm
import copy
import random
random.seed(5)

cand_path = "/egr/research-hlr/joslin/candidate.npy"
cand_data = np.load(cand_path, allow_pickle=True).item()

data_path = '/localscratch/zhan1624/CLIP/data_generation/raw_data/R2R_fake_test.json'
with open(data_path) as f_in:
    data = json.load(f_in)
print("---Finish Loading Data---")

landmark_path = "/localscratch/zhan1624/CLIP/data_generation/landmarks/all_output.json"
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

def generate_motion(motion_list):
    motion = random.choice(motion_list)
    return motion

def generate_landmarks(landmark_list):
    landmark = random.choice(landmark_list)
    return landmark

def unique(li1, li2):
    temp3 = []
    for element in li1:
        if element not in li2:
            temp3.append(element)
    return temp3
    

if __name__ == '__main__': 
    output_data = []
    for item in tqdm(data):
        new_data = item.copy()
        tmp_motion = copy.deepcopy(motions)
        scan = item['scan']
        path = item['path']
        generate_instr = []
        new_data["samples"] = {}
        heading = head_to_index(item['heading'])
        for view_id, view in enumerate(path):
            candidates = cand_data[scan][view]
            soft_neg_instr = []
            hard_neg_instr = []
            new_data["samples"][view] = {}
            if view_id == len(path)-1:
                landmark_list = landmarks[scan+"_"+view+"_"+str(heading)]
                direct_key = "STOP"
            else:
                tmp_landmark = []
                for each_can in candidates:
                    tmp_landmark.append(landmarks[scan+"_"+view+"_"+str(each_can['pointId'])])

                for can_id, each_can in enumerate(candidates):
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
                        break
                del tmp_landmark[can_id]
                all_landmark = set([i for item in tmp_landmark for i in item])
                soft_negative_landmark = list(set(landmark_list) & all_landmark) #common landmarks
                if len(soft_negative_landmark) >= len(landmark_list): #negative length less than the full length
                    soft_negative_landmark = []
                hard_negative_landmark = list(all_landmark - set(landmark_list)) # distinguish landmarks for candidate viewpoints 
                positive_landmark =  unique(landmark_list, soft_negative_landmark)# distinguish landmarks for next viewpoint
              

            heading = next_heading
            # positive generation
            assert len(positive_landmark) > 0 
            pos_motion = generate_motion(tmp_motion[direct_key])
            pos_landmark = positive_landmark[0]
            pos_tmp_instr = pos_motion+" "+pos_landmark
            tmp_motion[direct_key].remove(pos_motion)

            # negative generation
            neg_motion = copy.deepcopy(tmp_motion[direct_key])
            if soft_negative_landmark or hard_negative_landmark:
                if soft_negative_landmark:
                    for each_soft in soft_negative_landmark:
                        neg_m = generate_motion(neg_motion)
                        soft_neg_instr.append(neg_m+" "+each_soft)
                        neg_motion.remove(neg_m)

                while len(hard_negative_landmark)>0  and (len(soft_neg_instr)+len(hard_neg_instr))<3:
                    hard_land = generate_landmarks(hard_negative_landmark)
                    neg_m = generate_motion(neg_motion)
                    hard_neg_instr.append(neg_m+" "+hard_land)
                    hard_negative_landmark.remove(hard_land)
                    neg_motion.remove(neg_m)
            
            generate_instr.append(pos_tmp_instr)
            new_data["samples"][view]['pos_instr'] = pos_tmp_instr
            new_data["samples"][view]['neg_soft_instr'] = soft_neg_instr
            new_data["samples"][view]['neg_hard_instr'] = hard_neg_instr
        new_data['generate_instr'] = generate_instr
        output_data.append(new_data)
    with open("/localscratch/zhan1624/CLIP/data_generation/new_data/R2R_fake_test_new.json", 'w') as f_out:
        json.dump(output_data, f_out, indent=4)


        

              
