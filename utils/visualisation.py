import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def draw_confusion_matrix(matr, labels):
    df_cm = pd.DataFrame(matr, index=labels,
                         columns=labels)
    plt.figure(figsize=(10, 7))
    return sns.heatmap(df_cm, annot=True)


def show_image(image, title='img', wk=1):
    cv2.imshow(title, image)
    cv2.waitKey(wk)


def alpha_blend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim == 3 and mask.shape[-1] == 3:
        alpha = mask / 255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    blended = cv2.convertScaleAbs(img1 * (1 - alpha) + img2 * alpha)
    return blended
