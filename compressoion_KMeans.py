import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2.cv2 as cv2

colors = input("Number of colors: ")
layers = 4

image_original = mpimg.imread("monument.PNG") # RGBA, normalize data itself
# image_original = cv2.imread("monument.PNG") # RGB, do not normalize itself, we have to do it

# print(image_original)
img_data = (image_original).reshape(-1, layers)
KMeans = MiniBatchKMeans(int(colors)).fit(img_data)
kcolors = KMeans.cluster_centers_[KMeans.predict(img_data)]
image_compressed = np.reshape(kcolors, (image_original.shape))

fig, axarr = plt.subplots(nrows= 1, ncols=2)
axarr[0].set_title("Original")
axarr[0].imshow(image_original)
axarr[1].set_title("Compressed")
axarr[1].imshow(image_compressed)

axarr[0].axis('off')
axarr[1].axis('off')
fig.savefig("original+compressed_monument.PNG")
plt.show()
plt.close(fig)

