import os
import pandas as pd
import torch
from ssd import build_ssd
import cv2
from eval_utils import ImageInfer
# from PIL import Image

model = 'weights/bonecell_mean_0_0_0/ssd300_BONECELL_final.pth'
dataset_mean = (0, 0, 0) # works better than using imagenet mean..

def predict2(image, file_name, slices_per_axis):
    image.save('inputs/{}'.format(file_name))
    threshold = 0.5
    num_classes = 5
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
    net.eval()

    imgInfer = ImageInfer(img_dir='inputs/', img_fname=file_name, slices_per_axis=slices_per_axis)
    print('finished loading {}'.format(file_name))
    imgInfer.predict(net)
    print('finished predicting.')
    pred_w_nms_img, pred_w_nms_area = imgInfer.parse_results(threshold=threshold)
    print('num of rects: {}'.format(sum(imgInfer.post_nms_num_of_recs)))
    print('relative area: {}'.format(pred_w_nms_area))
    res = {
        'rects': sum(imgInfer.post_nms_num_of_recs),
        'relative_area': pred_w_nms_area,
    }
    resized_image = cv2.resize(pred_w_nms_img, (2000, 2000))
    imgInfer.deleteImg()

    # output_file_name = os.path.join('outputs/', 'pred_output_' + file_name + '_slice_{}.png'.format(slices_per_axis))
    # cv2.imwrite(output_file_name, resized_image)
    return resized_image, res
