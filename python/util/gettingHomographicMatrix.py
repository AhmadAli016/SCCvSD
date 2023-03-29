import cv2 as cv
import numpy as np
from PIL import Image

from .projective_camera import ProjectiveCamera
from .iou_util import IouUtil

# def get_homography_for_image(image_path):
# image = Image.open('001_AB_real_D (batchSize-7  - test_batch_7 - 30 epochs).png')
image_path = "001_AB_real_D (batchSize-7  - test_batch_7 - 30 epochs).png"

camera_data = np.asarray([640, 360, 3081.976880,
                            1.746393, -0.321347, 0.266827,
                            52.816224, -54.753716, 19.960425])

u, v, fl = camera_data[0:3]
rod_rot = camera_data[3:6]
cc = camera_data[6:9]

camera = ProjectiveCamera(fl, u, v, cc, rod_rot)

h = IouUtil.template_to_image_homography_uot(camera)
inv_h = np.linalg.inv(h)
im = cv.imread(image_path)
# im = Image.open('001_AB_real_D (batchSize-7  - test_batch_7 - 30 epochs).png')
assert im is not None

template_size = (115, 74);
warped_im = IouUtil.homography_warp(inv_h, im, template_size, (0, 0, 0))

    # return h
print(h)
print(warped_im)