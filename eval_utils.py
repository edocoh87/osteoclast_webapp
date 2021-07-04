import os
import numpy as np
import time
import torch
from torch.autograd import Variable
from PIL import Image
from data import BaseTransform, BoneCellInfer
from layers import box_utils
from ssd import build_ssd
import pickle
import cv2

dataset_mean = (0, 0, 0)

IDX_TO_CALC_AREA = 4

BONE_CELL_CLASSES_MAP = {
    'p': 1,
    'g': 2,
    '0_1': 3,
    '0_2': 4,
}

def parse_predictions(results, idx, threshold):
    curr_img_recs = []
    for cls, cls_idx in BONE_CELL_CLASSES_MAP.items(): #class
        curr_cls_bboxes = results[cls_idx][idx]
        for curr_box in curr_cls_bboxes:
            if curr_box[-1] >= threshold:
                curr_img_recs.append(curr_box.tolist() + [cls_idx])
    return curr_img_recs

def _slice(input_dir, inputfile, output_dir, k_w, k_h):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img = cv2.imread(os.path.join(input_dir, inputfile))
    im_h, im_w, channels = img.shape
    box_w = int(round(im_w/k_w))
    box_h = int(round(im_h/k_h))
    fname, ext = inputfile.split('.')
    im = Image.open(os.path.join(input_dir, inputfile))
    for _i, i in enumerate(range(0, im_w-1, box_w)):
        for _j, j in enumerate(range(0, im_h-1, box_h)):
            rect = (i, j, i+box_w, j+box_h)
            curr_fname = fname + '_{}_{}.'.format(_i+1, _j+1)
            a = im.crop(rect)
            a.save(os.path.join(output_dir, curr_fname + ext))

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
  
def test_net(save_folder, results_file_name, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(BONE_CELL_CLASSES_MAP)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, results_file_name)

    for i in range(num_images):
        # print('-'*100)
        # print(dataset.ids[i])
        im, gt, _, h, w = dataset.pull_item(i)
        x = Variable(im.unsqueeze(0)).float()
        if cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        # print(detections)
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= im_size
            boxes[:, 2] *= im_size
            boxes[:, 1] *= im_size
            boxes[:, 3] *= im_size
            # boxes[:, 0] *= w
            # boxes[:, 2] *= w
            # boxes[:, 1] *= h
            # boxes[:, 3] *= h

            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

class ImageInfer:
    def __init__(self, img_dir, img_fname, slices_per_axis):
        self.img_dir = img_dir
        self.img_fname = img_fname
        self.sliced_img_save_dir = self.img_fname[:-4] + '_slices_{}'.format(slices_per_axis)
        self.output_dir = os.path.join(self.img_dir, self.sliced_img_save_dir)
        self.output_dir_name_template = self.img_fname[:-4] + '_{}_{}.png'
        self.slices_per_axis = slices_per_axis
        self.num_of_recs = []
        self.post_nms_num_of_recs = []
        # self.results_file_name = None
        self.results_file_name = 'detection_results.pkl'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            _slice(self.img_dir, self.img_fname, self.output_dir, self.slices_per_axis, self.slices_per_axis)
        
        self.dataset = BoneCellInfer(root=self.output_dir, transform=BaseTransform(300, dataset_mean))

    def stitch_image(self, img_arr):
        img_2d_arr = []
        for j in range(self.slices_per_axis):
            curr_col = []
            for i in range(self.slices_per_axis):
                curr_col.append(img_arr[i*self.slices_per_axis + j])
            img_2d_arr.append(curr_col)
        cols = [cv2.vconcat(img_2d_arr[0])]
        for i in range(1, len(img_2d_arr)):
            cols.append(cv2.vconcat(img_2d_arr[i]))
        return cv2.hconcat(cols)
    
    def predict(self, net):
        test_net(
            save_folder=self.output_dir,
            results_file_name=self.results_file_name,
            net=net,
            cuda=False,
            dataset=self.dataset,
            transform=BaseTransform(net.size, dataset_mean),
            top_k=5,
            im_size=300,
            thresh=0.05)

    def parse_image(self, idx, curr_img_recs):
        bboxes = torch.Tensor(np.array([_box[:4] for _box in curr_img_recs]))
        scores = torch.Tensor(np.array([_box[4] for _box in curr_img_recs]))
        
        post_nms_img_recs = curr_img_recs
        if bboxes.shape[0] > 0:
            keep, count = box_utils.nms(bboxes, scores)
            keep = list(keep.numpy())
            if count < len(keep):
                post_nms_img_recs = [curr_img_recs[k] for k in keep]
        
        self.num_of_recs.append(len(curr_img_recs))
        self.post_nms_num_of_recs.append(len(post_nms_img_recs))

        gt_img, gt_area = self.dataset.draw_gt(idx, get_area=True)
        pred_img, pred_area = self.dataset.draw_pred(idx, pred_bbox=curr_img_recs, get_area=True)
        post_nms_pred_img, post_nms_pred_area = self.dataset.draw_pred(idx, pred_bbox=post_nms_img_recs, get_area=True)

        return (gt_img, gt_area), (pred_img, pred_area), (post_nms_pred_img, post_nms_pred_area)

    def parse_results(self, threshold=0.5):
        with open(os.path.join(self.output_dir, self.results_file_name), 'rb') as f:
            results = pickle.load(f)
        
        gt_img_arr = []
        pred_img_arr = []
        post_nms_pred_img_arr = []
        post_nms_pred_img_area_arr = []
        img_res_classes = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
        }
        for i in range(self.slices_per_axis):
            for j in range(self.slices_per_axis):
                idx = self.dataset.get_img_idx(self.output_dir_name_template.format(j+1, i+1))
                _curr_img_recs = parse_predictions(results, idx, threshold)
                curr_img_recs = [_ for _ in _curr_img_recs if _[5] == IDX_TO_CALC_AREA]
                for _ in curr_img_recs:
                    img_res_classes[_[5]] += 1
                (gt_img, gt_area), (pred_img, pred_area), (post_nms_pred_img, post_nms_pred_area) = self.parse_image(idx, curr_img_recs)
                gt_img_arr.append(gt_img)
                pred_img_arr.append(pred_img)
                post_nms_pred_img_arr.append(post_nms_pred_img)
                post_nms_pred_img_area_arr.append(post_nms_pred_area)
        img_area = np.mean(post_nms_pred_img_area_arr)

        return self.stitch_image(post_nms_pred_img_arr), img_area