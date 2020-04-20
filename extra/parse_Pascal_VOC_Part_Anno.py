import json
import numpy as np
import os
import scipy.io
from tqdm import tqdm


def get_bndbox_from_mask(bin_mask):
    y, x = np.where(np.array(bin_mask) > 0)
    return {'xmin': int(x.min()), 'ymin': int(y.min()), 'xmax': int(x.max()), 'ymax': int(y.max())}


def parse_pascal_part_mat_anno(data_dir, filename):
    """
    parse <data_dir>/<filename>.mat into json format
    PASCAL-Parts annotations dataset: http://roozbehm.info/pascal-parts/pascal-parts.html
    """
    data = scipy.io.loadmat('%s/%s.mat' % (data_dir, filename))['anno'][0][0]
    assert data[0][0] == filename
    
    parsed_anno = {'filename': filename, 'object': []}
    
    n_objects = len(data[1][0])
    for obj in data[1][0]:
        assert len(obj[0]) == 1
        obj_class = obj[0][0]
        if obj_class == 'table': # this class name is wrong in the pascal parts data annotations
            obj_class = 'diningtable'

        assert len(obj[1]) == 1 and len(obj[1][0]) == 1
        class_id = obj[1][0][0]

        obj_mask = np.array(obj[2])
        assert len(obj_mask.shape) == 2
        
        obj_anno = {'name': obj_class, 'bndbox': get_bndbox_from_mask(obj_mask), 'parts': []}

        if obj[3].shape == (0, 0): # no parts
            pass
        else:
            assert obj[3].shape[0] == 1
            n_parts = obj[3].shape[1]

            for part in obj[3][0]:
                assert len(part) == 2 and len(part[0]) == 1
                part_class = part[0][0]

                part_mask = np.array(part[1])
                assert len(part_mask.shape) == 2

                part_anno = {'name': part_class, 'bndbox': get_bndbox_from_mask(part_mask)}
                obj_anno['parts'].append(part_anno)
        
        parsed_anno['object'].append(obj_anno)
    
    return parsed_anno


if __name__ == '__main__':
    files = [f.name[:-4] for f in os.scandir('../data/VOCdevkit/VOC2010/Annotations_Part/') if f.name.endswith('.mat')]
    for filename in tqdm(files):
        parsed_anno = parse_pascal_part_mat_anno('../data/VOCdevkit/VOC2010/Annotations_Part/', filename)
        json.dump(parsed_anno, open('%s/%s.json' % ('../data/VOCdevkit/VOC2010/Annotations_Part_json/', filename), 'w'))