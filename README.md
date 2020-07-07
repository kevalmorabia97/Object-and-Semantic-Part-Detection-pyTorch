# Object-and-Semantic-Part-Detection-pyTorch
Joint detection of Object and its Semantic parts using Attention-based feature fusion for 2 Faster RCNN models. This project is done as a part of our CS543 Computer Vision Project at UIUC.<br>
[Link To ArXiv Pre-Print](https://arxiv.org/abs/2007.02419)

## Model Architecture:
We build our model on top of torchvision's Faster-RCNN model. Our model architecture is highly motivated from [this paper](https://link.springer.com/chapter/10.1007/978-3-030-20873-8_32) in that we replace the Relationship modeling and LSTM based feature fusion with an ```Attention-based feature fusion``` architecture.
We define a hyperparameter called ```fusion_thresh``` that decides which object and part proposals boxes are related to each other and should undergo fusion. ```fusion_thresh=0.9``` means that we consider those object and part boxes where their intersection area is atleast ```0.9*area_of_part```.
![](https://github.com/kevalmorabia97/Object-and-Semantic-Part-Detection-pyTorch/blob/master/extra/imgs/architecture.png)<br>
![](https://github.com/kevalmorabia97/Object-and-Semantic-Part-Detection-pyTorch/blob/master/extra/imgs/feature_fusion.png)

More details in [Project_Report.pdf](https://github.com/kevalmorabia97/Object-and-Semantic-Part-Detection-pyTorch/blob/master/Project_Report.pdf) file and [Video Presentation](https://drive.google.com/open?id=1cdqaQUN77hxnAd_xIfeRv47RRTU5W9wX).

## Dataset Info:
**Dataset:** [PASCAL VOC 2010 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/index.html#devkit)<br>
**Annotations from:** [PASCAL Parts dataset](http://roozbehm.info/pascal-parts/pascal-parts.html)
<br>Part Annotations are preprocessed from  \*.mat format to \*.json using ```scipy.io.loadmat``` module and segmentation masks are used to get corresponding bounding box localizations. [Link to parsed part annotations](https://drive.google.com/drive/folders/1sF1NY0VygsvoGvRkp90e3x5zYgx7RRKc?usp=sharing)

```
Directory structure for data is as follows:

data
└── VOCdevkit
    └── VOC2010
        ├── Annotations_Part_json
        │   └── *.json
        ├── Classes
        │   └── *class2ind.txt
        ├── ImageSets
        │   └── Main
        │       └── *train/val.txt
        └── JPEGImages
            └── *.jpg
```

**Dataset Preprocessing:**
<br>Our goal is to show that having part information can improve object detection performance, and vice versa.
<br>Some classes e.g. ```boat``` do not have part annotations. So we discard them from our ```vehicles_train/val.txt``` file.
<br>Some images don't have part annotations for any objects. So we discard them from our corresponding ```train/val.txt``` file.
<br>All the above kinds of removed images were about 0.5% only.
<br>No images have been removed from combined train/val/trainval files. They are only removed from ```animals/indoor/person/vehicles train/val``` files. So running the part detection model on entire dataset for all classes would result in lots of samples where there will be no part annotations.

<br>In total, there were ```166 part classes```. We coarse-grained these part annotations by merging multiple parts into a single class. For example ```FACE``` is the new part class combining ```[beak, hair, head, nose, lear, lebrow, leye, mouth, rear, rebrow, reye]```. After all merging, number of part classes has reduced to 19 which is present in ```Classes/part_mergedclass2ind.txt```.
<br> Below image shows part classes before and after merging. More example can be found in ```data/VOCdevkit/VOC2010/example_merged_part_images```.

Before Merging Part Classes|After merging Part Classes
:-------------------------:|:-------------------------:
![](https://github.com/kevalmorabia97/Object-and-Semantic-Part-Detection-pyTorch/blob/master/data/VOCdevkit/VOC2010/example_merged_part_images/2008_000217_allparts.jpg)  |  ![](https://github.com/kevalmorabia97/Object-and-Semantic-Part-Detection-pyTorch/blob/master/data/VOCdevkit/VOC2010/example_merged_part_images/2008_000217_mergedparts.jpg)
![](https://github.com/kevalmorabia97/Object-and-Semantic-Part-Detection-pyTorch/blob/master/data/VOCdevkit/VOC2010/example_merged_part_images/2008_000112_allparts.jpg)  |  ![](https://github.com/kevalmorabia97/Object-and-Semantic-Part-Detection-pyTorch/blob/master/data/VOCdevkit/VOC2010/example_merged_part_images/2008_000112_mergedparts.jpg)

## Running the code (with default parameters):
For training a single model for animal object detection:<br>
```python3 main.py -e 15 --use_objects -tr animals_train -val animals_val -cf animals_object_class2ind```

For training a single model for animal part detection:<br>
```python3 main.py -e 15 --use_parts -tr animals_train -val animals_val -cf animals_part_mergedclass2ind```

For training the joint model for simultaneous animal object and part detection:<br>
```python3 main_joint.py -e 15 -ft 0.9```

## Results:
From our limited experiments for Animal Object ```(bird, cat, cow, dog, horse, sheep)``` and Part ```(face, leg, neck, tail, torso, wings)``` Detection, we find that the Attention-based Joint Detection model gives improvement for Part classes in terms of ```mean Average Precision @IoU=0.5```.

| Model | Object Detection mAP@IoU=0.5 | Part Detection mAP@IoU=0.5 |
| ------------- | ------------- | ------------- |
| Single Object Detection Model  | 87.2  | --  |
| Single Part Detection Model | --  | 51.3  |
| Joint Object and Part Detection | **87.5**  | **52.0**  |


| Object Detections from Joint Model | Part Detections from Joint Model |
| ------------- | ------------- |
| ![](https://github.com/kevalmorabia97/Object-and-Semantic-Part-Detection-pyTorch/blob/master/extra/imgs/2008_000080_obj.jpg)  | ![](https://github.com/kevalmorabia97/Object-and-Semantic-Part-Detection-pyTorch/blob/master/extra/imgs/2008_000080_part.jpg) |
| ![](https://github.com/kevalmorabia97/Object-and-Semantic-Part-Detection-pyTorch/blob/master/extra/imgs/2008_000765_obj.jpg)  | ![](https://github.com/kevalmorabia97/Object-and-Semantic-Part-Detection-pyTorch/blob/master/extra/imgs/2008_000765_part.jpg) |

## Requirements:
```
The code has been tested on the following requirements:

numpy==1.18.2
pycocotools==2.0.0
Pillow==6.2.2
torch==1.3.0
torchvision==0.4.1
tqdm==4.4.11
```

## Contributors:
1. Keval Morabia ([kevalmorabia97](https://github.com/kevalmorabia97/))
2. Jatin Arora ([jatinarora2702](https://github.com/jatinarora2702))
3. Tara Vijaykumar ([tara-vijaykumar](https://github.com/tara-vijaykumar))

## Cite
If you find this useful in your research, please cite our ArXiv pre-print:
```
@misc{morabia2020attentionbased,
    title={Attention-based Joint Detection of Object and Semantic Part},
    author={Keval Morabia and Jatin Arora and Tara Vijaykumar},
    year={2020},
    eprint={2007.02419},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```


## Abstract:
In this paper, we address the problem of joint detection of objects like dog and its semantic parts like face, leg, etc. Our model is created on top of two Faster-RCNN models that share their features to perform a novel Attention-based feature fusion of related Object and Part features to get enhanced representations of both. These representations are used for final classification and bounding box regression separately for both models. Our experiments on the PASCAL-Part 2010 dataset show that joint detection can simultaneously improve both object detection and part detection in terms of mean Average Precision (mAP) at IoU=0.5. 
