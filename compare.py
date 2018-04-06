from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "C:\Users\Mypc\Desktop\Project\images")
args = vars(ap.parse_args())

index = {}
images = {}

for imagePath in glob.glob(args["dataset"] + "\*.jpg"):
	
	filename = imagePath[imagePath.rfind("\\") + 1:]
	image = cv2.imread(imagePath)
	images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
	hist = cv2.normalize(hist).flatten()
	index[filename] = hist



SCIPY_METHODS = (
	("Euclidean", dist.euclidean),
	("Manhattan", dist.cityblock),
	("Mahanalobis", dist.chebyshev))


for (methodName, method) in SCIPY_METHODS:
	
	results = {}

	
	for (k, hist) in index.items():
		
		d = method(index["m802.jpg"], hist)
		results[k] = d

	
	results = sorted([(v, k) for (k, v) in results.items()])

	
	fig = plt.figure("Query")
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(images["m802.jpg"])
	plt.axis("off")

	
	fig = plt.figure("Results: %s" % (methodName))
	fig.suptitle(methodName, fontsize = 20)

        B=0.2
        cnt=0
	for (i, (v, k)) in enumerate(results):
                if v <= B:
                        cnt=cnt+1
	for (i, (v, k)) in enumerate(results):
		if v <= B:
                        ax = fig.add_subplot(1, cnt+1, i+1)
                        ax.set_title("%.2f" % (v))
                        plt.imshow(images[k])
                        plt.axis("off")

plt.show()

