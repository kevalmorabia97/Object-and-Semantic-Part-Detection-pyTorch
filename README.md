# Object-and-Semantic-Part-Detection-pyTorch
Joint detection of Object and its Semantic parts using Faster RCNN model.

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
:-------------------------:|:-------------------------:
![](https://github.com/kevalmorabia97/Object-and-Semantic-Part-Detection-pyTorch/blob/master/data/VOCdevkit/VOC2010/example_merged_part_images/2008_000112_allparts.jpg)  |  ![](https://github.com/kevalmorabia97/Object-and-Semantic-Part-Detection-pyTorch/blob/master/data/VOCdevkit/VOC2010/example_merged_part_images/2008_000112_mergedparts.jpg)

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
