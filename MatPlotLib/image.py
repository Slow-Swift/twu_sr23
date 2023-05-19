import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

img = np.asarray(Image.open('Images/Map.png'))
imgplot = plt.imshow(img)
plt.show()