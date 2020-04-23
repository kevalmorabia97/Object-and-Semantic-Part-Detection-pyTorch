This directory contains various files for training on different sets of object/part classes

Many fine-grained part classes are combined into single class for more coarse-grained parts
For example: "FACE" is the new part class combining [beak, hair, head, nose, lear, lebrow, leye, mouth, rear, rebrow, reye]
More info can be found in part_classes_merged_info.txt
Note that some part classes are ignored to simplify the dataset, for example mirror part is removed.
List of all total 166 part classes before removing any part can be found in part_class2ind.txt

After all merging, number of part classes has reduced to 19 which is present in part_mergedclass2ind.txt

There are 4 super-categories in the entire dataset: Animals, Indoor, Person, Vehicles. For these super-categories also, the object/part classes are available if you want to train a model for let's say animal detection.

Annotations for all 166 fine-grained classes can be found in Annotations_Part_json directory.
Annotations for all 19 coarse-grained merged part classes can be found in Annotations_Part_json_merged_part_classes directory.