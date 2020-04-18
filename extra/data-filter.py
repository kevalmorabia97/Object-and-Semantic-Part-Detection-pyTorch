
def read_file(filename):
	# annotation_dir = "./data/VOCdevkit/VOC2010/Annotations_Part_json/"
	data = {"-1": [], "0": [], "1": []}
	with open(filename, "r") as f:
		for line in f:
			splits = line.split()
			data[splits[1].strip()].append(splits[0].strip())
	return data


def collate_positive_samples(classnames, rootclassname, outpath):
	data_train = []
	data_val = []

	root = "./data/VOCdevkit/VOC2010/ImageSets/Main/"
	
	for classname in classnames:
		d1 = read_file(root + classname + "_train.txt")
		d2 = read_file(root + classname + "_val.txt")

		# validatate that there is no overlap between the train and validation sets
		for x in d1["1"]:
			if x in d2["1"]:
				print("error: ", x)

		data_train += d1["1"]
		data_val += d2["1"]

	f = open(outpath + rootclassname + "_train.txt", "w")
	for d in set(data_train):
		f.write(d + "\n")

	f = open(outpath + rootclassname + "_val.txt", "w")
	for d in set(data_val):
		f.write(d + "\n")


if __name__ == "__main__":
	outpath = "./category-wise/"
	# collate_positive_samples(["tvmonitor", "pottedplant", "bottle", "chair", "diningtable", "sofa"], "indoor", outpath)
	# collate_positive_samples(["train", "aeroplane", "car", "motorbike", "bicycle", "bus", "boat"], "vehicle", outpath)
	
	collate_positive_samples(["horse", "dog", "cat", "bird", "sheep", "bus", "cow"], "animals", outpath)
	collate_positive_samples(["tvmonitor", "pottedplant", "bottle"], "indoor", outpath) # removing those which do not have granunal part information available
	collate_positive_samples(["train", "aeroplane", "car", "motorbike", "bicycle", "bus"], "vehicle", outpath)  # removing those which do not have granunal part information available
	collate_positive_samples(["person"], "person", outpath)
