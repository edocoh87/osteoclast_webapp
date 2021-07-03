import os
import pandas as pd
import torch
from ssd import build_ssd
import cv2
from eval_utils import ImageInfer
# from PIL import Image

model = 'weights/bonecell_mean_0_0_0/ssd300_BONECELL_final.pth'
dataset_mean = (0, 0, 0) # works better than using imagenet mean..

    # def remove_prefix(text, prefix):
    #     if text.startswith(prefix):
    #         return text[len(prefix):]
    #     return text  # or whatever
    #
    # # count_per_image = {}
    # # img_base_dir = '/Users/edock/Work/bonecell_infer/SAPIR_LABELED_TEST_copy/'
    # # img_base_dir = '/Users/edock/Work/bonecell_infer/zamzam_figures/FIGURE_2_and_3/'
    # img_base_dir = '/Users/edock/Work/bonecell_infer/Zamzam/fwdosteoclasts/'
    # # img_base_dir = '/Users/edock/Work/bonecell_infer/test_data_class4/'
    #
    # file_list = []
    #
    # for subdir, dirs, files in os.walk(img_base_dir):
    #     for file in files:
    #         if file.endswith('.png'):
    #             # file_list.append(file)
    #             fpath = os.path.join(subdir, file)
    #             file_list.append(remove_prefix(fpath, img_base_dir))
    #         else:
    #             print('file: {} not included...'.format(os.path.join(subdir, file)))
    #
    # for _ in file_list:
    #     print(_)
    # # exit()
    #
    # # print(file_list)
    # # output_dir_base = '/Users/edock/Work/bonecell_infer/zamzam_FIGURE_2_and_3_output/'
    # output_dir_base = '/Users/edock/Work/bonecell_infer/test_data_class4_output/'
    #
    # # img_base_dir = '/Users/edock/Work/bonecell_infer/FIGURE1/EXP.1/'
    # slices_per_axis = 3
    # # output_dir = os.path.join(img_base_dir, 'new_output_slice_{}'.format(slices_per_axis))
    # threshold = 0.5
    # num_classes = 5
    # net = build_ssd('test', 300, num_classes) # initialize SSD
    # # net.load_state_dict(torch.load(args.trained_model))
    # net.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
    # net.eval()
    # print('Finished loading model!')
    #
    #
    # cells_per_file = []
    # area_per_file = []
    # for file in file_list:
    #     if '/' in file:
    #         _file = '/'.join(file.split('/')[:-1])
    #         fname = file.split('/')[-1]
    #         # fname = file.split('/')[-1] + '.png'
    #     else:
    #         _file = ''
    #         fname = file
    #     img_dir = os.path.join(img_base_dir, _file)
    #
    #     imgInfer = ImageInfer(img_dir=img_dir, img_fname=fname, slices_per_axis=slices_per_axis)
    #     print('finished loading {}'.format(file))
    #     imgInfer.predict(net)
    #     print('finished predicting.')
    #     pred_w_nms_img, pred_w_nms_area = imgInfer.parse_results(threshold=threshold)
    #     # gt_img, pred_img, pred_w_nms_img = imgInfer.parse_results()
    #     # pred_w_nms_img = imgInfer.parse_results()
    #     # print(sum(imgInfer.num_of_recs))
    #     # print(sum(imgInfer.post_nms_num_of_recs))
    #     cells_per_file.append(sum(imgInfer.post_nms_num_of_recs))
    #     area_per_file.append(pred_w_nms_area)
    #     resized_image = cv2.resize(pred_w_nms_img, (2000, 2000))
    #     if not os.path.exists(os.path.join(output_dir_base, _file)):
    #         os.makedirs(os.path.join(output_dir_base, _file))
    #     cv2.imwrite(os.path.join(output_dir_base, _file, 'pred_output_' + fname + '_slice_{}.png'.format(slices_per_axis)), resized_image)
    #     # resized_image = cv2.resize(pred_w_nms_img, (1000, 1000))
    #     # cv2.imshow('image', resized_image)
    #     # cv2.waitKey(0)
    #     # im = Image.open(file + '.gif')
    #     # im.save(file + ".png")
    #     # im.show()
    # df = pd.DataFrame({'files': file_list, 'counts': cells_per_file, 'area': area_per_file})
    # df.to_csv(os.path.join(output_dir_base, 'results_thrs_{}.csv'.format(slices_per_axis, threshold)))

def predict2(image, file_name, slices_per_axis):
    image.save('inputs/{}'.format(file_name))
    # slices_per_axis = 3
    threshold = 0.5
    num_classes = 5
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    # net.load_state_dict(torch.load(args.trained_model))
    net.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
    net.eval()

    imgInfer = ImageInfer(img_dir='inputs/', img_fname=file_name, slices_per_axis=slices_per_axis)
    print('finished loading {}'.format(file_name))
    imgInfer.predict(net)
    print('finished predicting.')
    pred_w_nms_img, pred_w_nms_area = imgInfer.parse_results(threshold=threshold)
    # gt_img, pred_img, pred_w_nms_img = imgInfer.parse_results()
    # pred_w_nms_img = imgInfer.parse_results()
    # print(sum(imgInfer.num_of_recs))
    # print(sum(imgInfer.post_nms_num_of_recs))
    print('num of rects: {}'.format(sum(imgInfer.post_nms_num_of_recs)))
    print('relative area: {}'.format(pred_w_nms_area))
    res = {
        'rects': sum(imgInfer.post_nms_num_of_recs),
        'relative_area': pred_w_nms_area,
    }
    # cells_per_file.append(sum(imgInfer.post_nms_num_of_recs))
    # area_per_file.append(pred_w_nms_area)
    resized_image = cv2.resize(pred_w_nms_img, (2000, 2000))
    # if not os.path.exists(os.path.join(output_dir_base, _file)):
    #     os.makedirs(os.path.join(output_dir_base, _file))
    # cv2.imwrite(os.path.join(output_dir_base, _file, 'pred_output_' + fname + '_slice_{}.png'.format(slices_per_axis)),
    output_file_name = os.path.join('outputs/', 'pred_output_' + file_name + '_slice_{}.png'.format(slices_per_axis))
    cv2.imwrite(output_file_name, resized_image)
    return resized_image, res

# from PIL import Image
# # image = Image.open('/Users/edock/Work/bonecell_infer/Zamzam/fwdosteoclasts/cntl-aa-4a.png')
# image = Image.open('/Users/edock/Work/bonecell_infer/Zamzam/zamzam_figures/FIGURE_2_and_3/21.10.18/ARA (1).png')
# fname = 'ARA (1).png'
# predict2(image, fname)