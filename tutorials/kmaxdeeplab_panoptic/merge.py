import os
import json
from tqdm import tqdm


train_path=YOUR_JSON_FILE_PATH
with open(train_path,'r') as f:
    train_panseg=json.load(f)

coconut_b_path=YOUR_COCONUT_B_PATH
with open(coconut_b_path,'r') as f:
    unlabel_panseg=json.load(f)

out_panseg=train_panseg.copy()
new_img_list=unlabel_panseg['images']
new_annos=unlabel_panseg['annotations']

    
out_panseg['images'].extend(new_img_list)
out_panseg['annotations'].extend(new_annos)

with open('./annotations/panoptic_train2017.json','w') as f:
    json.dump(out_panseg, f, indent=4)