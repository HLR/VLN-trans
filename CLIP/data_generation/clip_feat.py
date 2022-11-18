
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'

import torch
import clip
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
images_path = "/egr/research-hlr/joslin/generated_images/"
labe_path = "faster_obj.txt"
scan_list = os.listdir(images_path)

### images
all_images = {}
for each_scan in tqdm(scan_list):
    scan_id = each_scan[:-4]
    scan_images = np.load(images_path+each_scan, allow_pickle=True).item()
    for view in scan_images[scan_id].keys():
        for id, value in scan_images[scan_id][view].items():
            all_images[scan_id+"_"+view+"_"+str(id)] = value
print("------Finish reading all images-----")


### labels
label_list = []
with open(labe_path) as f_read:
    for obj in f_read:
        label_list.append(obj.strip())

### model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in label_list]).to(device)


output = {}
for each_img_key, each_img_value in tqdm(all_images.items()):
    tmp_label = []
    image = preprocess(Image.fromarray(each_img_value)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)
    for each_i in indices:
        tmp_label.append(label_list[each_i])
    output[each_img_key] = tmp_label

with open('output6.json','w') as f_out:
    json.dump(output, f_out, indent=4)

print("------Work Done-----")
#bash 1; 0:10; output1
#bash 2; 10:20; output2
#bash 3; 20:30; output3
#bash 4; 30:40; output4
#bash 5; 40:50; output5
#bash 6; 50:60; output6
#bash 7; 60:70; output7
#bash 8; 70:80; output8
#bash 9; 80:90; output9

### merge all parallel generated json files
def merge():
    out_path = "/localscratch/zhan1624/CLIP/data_process/outputs"
    out_files = os.listdir(out_path)

    all_data = {}
    for file in tqdm(out_files):
        with open(out_path+"/"+file) as f_in:
            each_data = json.load(f_in)
        all_data.update(each_data)

    with open("/localscratch/zhan1624/CLIP/data_process/outputs/all_output.json", 'w') as f_out:
        json.dump(all_data, f_out, indent=4)





def check():
    """
    check whether there are missing keys
    """
    import json
    import pickle
    from tqdm import tqdm
    with open("/egr/research-hlr/joslin/img_features/objects/pano_object_class.pkl", "rb") as f_check:
        check = pickle.load(f_check)

    all_images = {}
    for scan_key, scan_value in tqdm(check.items()):
        for view_key, view_value in scan_value.items():
            for index_key, index_value in view_value.items():
                all_images[scan_key+"_"+view_key+"_"+str(index_key)] = index_value


    with open("/localscratch/zhan1624/CLIP/data_process/outputs/all_output.json") as f_in:
        all_dict = json.load(f_in)

    if all_images.keys() == all_dict.keys():
        print("Good Job")