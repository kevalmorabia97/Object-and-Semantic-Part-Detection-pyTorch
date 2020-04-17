import json
import os
import numpy as np


def print_class_distribution(setname, files):
	print("getting distribution of labels across classes and parts in the dataset")

	fine_parts_map = dict()
	parts_map = dict()
	obj_map = dict()
	
	# all_files = [f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))]
	for index, file in enumerate(files):
		data = json.load(open(file, "r"))
		
		for obj in data["object"]:
			obj_name = obj["name"]

			if obj_name in obj_map:
				obj_map[obj_name] += 1
			else:
				obj_map[obj_name] = 1
			
			if "parts" in obj:
				for part in obj["parts"]:
					part_name = part["name"]
					
					if part_name in parts_map:
						parts_map[part_name] += 1
					else:
						parts_map[part_name] = 1	
					
					full_name = obj_name + "." + part_name

					if full_name in fine_parts_map:
						fine_parts_map[full_name] += 1
					else:
						fine_parts_map[full_name] = 1	

		if index % 1000 == 0:
			print("index: ", index)
	
	with open(setname + "-obj-distribution.txt", "w") as f:
		for key in obj_map:
			f.write(key + ": " + str(obj_map[key]) + "\n")

	with open(setname + "-fine-parts-distribution.txt", "w") as f:
		for key in fine_parts_map:
			f.write(key + ": " + str(fine_parts_map[key]) + "\n")

	with open(setname + "-parts-distribution.txt", "w") as f:
		for key in parts_map:
			f.write(key + ": " + str(parts_map[key]) + "\n")


if __name__ == "__main__":
	root = "data/VOCdevkit/VOC2010/"
	image_dir = "%s/JPEGImages/" % root
	annotation_dir = "%s/Annotations_Part_json" % root
	
	# image_set = "train"
	# splits_file = '%s/ImageSets/Main/%s.txt' % (root, image_set)
	# file_names = np.loadtxt(splits_file, dtype=str)
	# annotations = ['%s/%s.json' % (annotation_dir, x) for x in file_names]
	# print_class_distribution(image_set, annotations)

	# image_set = "val"
	# splits_file = '%s/ImageSets/Main/%s.txt' % (root, image_set)
	# file_names = np.loadtxt(splits_file, dtype=str)
	# annotations = ['%s/%s.json' % (annotation_dir, x) for x in file_names]
	# print_class_distribution(image_set, annotations)

	image_set = "animals_train"
	splits_file = '%s/ImageSets/Main/%s.txt' % (root, image_set)
	file_names = np.loadtxt(splits_file, dtype=str)
	annotations = ['%s/%s.json' % (annotation_dir, x) for x in file_names]
	print_class_distribution(image_set, annotations)
