import os,yaml,shutil
from tqdm import tqdm

mask_list_dict = { 'face':['_skin','_l_ear','_l_eye','_l_brow',\
                '_neck','_nose','_r_ear','_r_eye','_r_brow','_whole_mouth','_hair'],
                                'building':['_building','_plant','_sky','_road'],
                                'mountain':['_mountain','_plant','_sky','_water'],
                                'horse':['']}

root = 'datasets/all_in/class'
# for label in mask_list_dict.keys():
for label in os.listdir(root):
    list_path = os.path.join(root,label,'list.yaml')
    if os.path.exists(list_path):
        os.remove(list_path)
    img_dict = {}
    for img_name in tqdm(os.listdir(os.path.join(root,label,'images'))):
        img_dict[img_name] = []
        for mask_name in mask_list_dict[label]:
            A_name = img_name[:-4] + mask_name + img_name[-4:]
            path = os.path.join(root,label, 'mask',A_name) 
            if os.path.exists(path):
                img_dict[img_name].append(mask_name)
                if mask_name == '_skin':
                    img_dict[img_name] += ['_skin','_skin','_skin','_skin']
    fp = open(os.path.join(root,label,'list.yaml'),'w')
    fp.write(yaml.dump(img_dict))
    fp.close()
    
