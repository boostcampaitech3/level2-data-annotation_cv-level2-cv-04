import os.path as osp
import json
import os
import copy
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-jl','--json_data_list', nargs='*',
    default= ["../input/data/1_ICDAR17_Korean/ufo/train.json", 
    "../input/data/2_ai_stage/ufo/train.json"])

arg = parser.parse_args()

ret_json={"images":{}}

for json_ in arg.json_data_list:
    with open(json_,'r') as f:
        json_data=json.load(f)

    print(len(json_data['images']))
    data_root = json_.split('/')[3] + '/images/'
    keys = list(json_data['images'].keys())
    for k in keys:
        json_data['images'][data_root+k] = json_data['images'].pop(k)
    print(len(json_data['images']))
    
    ret_json['images'].update(json_data['images'])
print(len(ret_json['images']))


anno = ret_json['images']

anno_temp = copy.deepcopy(anno)

count = 0
count_normal = 0
count_none_anno = 0

for img_name, img_info in tqdm(anno.items()) :
    if img_info['words'] == {} :
        del(anno_temp[img_name])
        count_none_anno += 1
        continue
    for obj, obj_info in img_info['words'].items() :
        # revised 버전일 경우 여기를 지워주세요
        #anno_temp[img_name]['words'][obj]['illegibility'] = False
        
        if len(img_info['words'][obj]['points']) == 4 :
            count_normal += 1
            continue
        # 폴리곤 수정시에는 여기 부분을 수정해주시면 됩니다!!
        # 다음 예제는 polygon이 넘치거나 모자를 경우 해당 폴리곤을 object를 삭제처리
        elif len(img_info['words'][obj]['points']) < 4 :
            del(anno_temp[img_name]['words'][obj])
            if anno_temp[img_name]['words'] == {} :
                del(anno_temp[img_name])
                count_none_anno += 1
                continue
        else :
            # 현동님의 기여로 만들어진 부분
            over_polygon_temp = copy.deepcopy(anno_temp[img_name]['words'][obj])
            over_poly_region = copy.deepcopy(over_polygon_temp)
            over_poly_region['points'] = []
            for index in range(len(img_info['words'][obj]['points'])//2 -1):
                over_poly_region['points'].append(over_polygon_temp['points'][index])
                over_poly_region['points'].append(over_polygon_temp['points'][index+1])
                over_poly_region['points'].append(over_polygon_temp['points'][-index-1])
                over_poly_region['points'].append(over_polygon_temp['points'][-index])
                anno_temp[img_name]['words'][obj+f'{index+911}'] = copy.deepcopy(over_poly_region) #911 현동님 생일 >_<
                over_poly_region['points'] = []
            del anno_temp[img_name]['words'][obj]
            # 폴리곤을 사각단위로 잘라서 각 부분을 word의 영역으로 사용하는 코드입니다.
            if anno_temp[img_name]['words'] == {} :
                del(anno_temp[img_name])
                count_none_anno += 1
                continue
            count += 1
            
print(f'normal polygon count : {count_normal}')
print(f'deleted {count} over polygon')

anno = {'images': anno_temp}


with open('output.json','w') as f:
    json.dump(anno,f,indent='\t')