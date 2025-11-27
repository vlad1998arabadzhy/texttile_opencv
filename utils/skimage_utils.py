from skimage.color import label2rgb, rgb2gray, rgba2rgb
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import footprint_rectangle, closing
from skimage.segmentation import clear_border

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches




def label_image(path):
    #Load
    img = imread(path)

    img = rgba2rgb(img)

    #Grayscaling
    img = rgb2gray(img)

    thresh = threshold_otsu(img)
    bw=closing(img>thresh, footprint_rectangle((3,3)))

    cleared =clear_border(bw)

    labeled_image = label(cleared)
    image_label_overlay = label2rgb(labeled_image, image=img, bg_label=0)
    fig, ax = plt.subplots()
    ax.imshow(image_label_overlay)

    for region in regionprops(labeled_image):
        if region.area > 100:
            minr, minc , maxr, maxc = region.bbox
            rect = mpatches.Rectangle(
                (minc, minr),
                (maxc-minc),
                (maxr-minr),
                fill=False,
                edgecolor='red',
                linewidth=2
            )

            ax.add_patch(rect)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    return

