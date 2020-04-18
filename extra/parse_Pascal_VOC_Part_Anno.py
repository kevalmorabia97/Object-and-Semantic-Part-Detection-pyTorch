import json
import numpy as np
import os
from tqdm import tqdm

from mat4py import loadmat


def get_bndbox_from_mask(bin_mask):
    y, x = np.where(np.array(bin_mask) > 0)
    return {'xmin': int(x.min()), 'ymin': int(y.min()), 'xmax': int(x.max()), 'ymax': int(y.max())}


def parse_PASCAL_PARTS_Anno_for_Detection(data_dir, output_dir):
    """
    parse `data_dir`/*.mat files and store them at `output_dir`/*.json
    PASCAL-Parts annotations dataset: http://roozbehm.info/pascal-parts/pascal-parts.html
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_names = [f.name[:-4] for f in os.scandir(data_dir) if f.name.endswith('.mat')]
    print('%d .mat file(s) found' % len(file_names))
    
    for file_name in tqdm(file_names):
        parsed_anno = {'filename': file_name, 'object': []}

        try:
            data = loadmat('%s/%s.mat' % (data_dir, file_name))['anno']['objects']
        except:
            # Doesn't have part annotation. Use PASCAL Annotations .xml files for these images
            # See https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py -> parse_voc_xml()
            continue
        
        if type(data['class']) != list:
            data['class'] = [data['class']]
            data['mask'] = [data['mask']]
            data['parts'] = [data['parts']]
        for obj_idx in range(len(data['class'])):
            obj_class = data['class'][obj_idx]
            obj_anno = {'name': obj_class, 'bndbox': get_bndbox_from_mask(data['mask'][obj_idx]), 'parts': []}
            
            if type(data['parts'][obj_idx]['part_name']) != list:
                data['parts'][obj_idx]['part_name'] = [data['parts'][obj_idx]['part_name']]
                data['parts'][obj_idx]['mask'] = [data['parts'][obj_idx]['mask']]
            for part_idx in range(len(data['parts'][obj_idx]['part_name'])):
                part_class = data['parts'][obj_idx]['part_name'][part_idx]
                part_anno = {'name': part_class, 'bndbox': get_bndbox_from_mask(data['parts'][obj_idx]['mask'][part_idx])}
                obj_anno['parts'].append(part_anno)

            parsed_anno['object'].append(obj_anno)

        json.dump(parsed_anno, open('%s/%s.json' % (output_dir, file_name), 'w'))
    print('Done! Parsed json files saved to %s' % output_dir)


if __name__ == '__main__':
    parse_PASCAL_PARTS_Anno_for_Detection('../data/VOCdevkit/VOC2010/Annotations_Part/', 'Annotations_Part_json/')