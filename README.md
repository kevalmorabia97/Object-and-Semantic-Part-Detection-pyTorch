# Object-and-Semantic-Part-Detection-pyTorch
Joint detection of Object and its Semantic parts using Faster RCNN model.

**Dataset:** [PASCAL VOC 2010 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/index.html#devkit)<br>
**Annotations from:** [PASCAL Parts dataset](http://roozbehm.info/pascal-parts/pascal-parts.html)
<br>Part Annotations are preprocessed from  \*.mat format to \*.json using ```scipy.io.loadmat``` module and segmentation masks are used to get corresponding bounding box localizations.
```
Directory structure for data is as follows:

data
└── VOCdevkit
    └── VOC2010
        ├── Annotations_Part_json
        │   └── *.json
        ├── ImageSets
        │   └── Main
        └── JPEGImages
            └── *.jpg
```

```
The code has been tested on the following requirements:

numpy==1.18.2
pycocotools==2.0.0
Pillow==6.2.2
torch==1.3.0
torchvision==0.4.1
tqdm==4.4.11
```