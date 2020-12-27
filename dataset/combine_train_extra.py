import pickle
import shutil 
from tqdm import tqdm as tqdm

with open('./train_annotations_processed.pkl' , 'rb') as f:
    train_annotations = pickle.load(f)
with open('./extra_annotations_processed.pkl' , 'rb') as f:
    extra_annotations = pickle.load(f)

final_annotations = train_annotations.copy()

for i in tqdm(train_annotations.keys()):
    train_src_path =  "./train/" + i
    train_target_path =  "./combined/" + i
    shutil.copyfile(train_src_path,train_target_path)

for i in tqdm(extra_annotations.keys()):
    new_img_index = len(train_annotations) + int(i.split('.')[0])
    new_img_abs_path = str(new_img_index) + '.' + i.split('.')[1]
    final_annotations[new_img_abs_path] = extra_annotations[i]
    extra_src_path =  "./extra/" + i
    extra_target_path =  "./combined/" + new_img_abs_path
    shutil.copyfile(extra_src_path,extra_target_path)

import pickle
with open('./combined_train_extra_annotations_processed.pkl' , 'wb') as f:
    pickle.dump(final_annotations,f)