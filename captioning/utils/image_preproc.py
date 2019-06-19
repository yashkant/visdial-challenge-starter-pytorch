import cv2
import numpy as np
import torch
import requests
from PIL import Image
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.structures.image_list import to_image_list


def get_actual_image(image_path):
    if image_path.startswith('http'):
        path = requests.get(image_path, stream=True).raw
    else:
        path = image_path

    return path


def image_transform(image_path):
    path = get_actual_image(image_path)

    img = Image.open(path)
    im = np.array(img).astype(np.float32)
    im = im[:, :, ::-1]
    im -= np.array([102.9801, 115.9465, 122.7717])
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(800) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > 1333:
        im_scale = float(1333) / float(im_size_max)
    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR
    )
    img = torch.from_numpy(im).permute(2, 0, 1)
    return img, im_scale


def process_feature_extraction(output,
                               im_scales,
                               feat_name='fc6',
                               conf_thresh=0.2):
    batch_size = len(output[0]["proposals"])
    n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
    score_list = output[0]["scores"].split(n_boxes_per_image)
    score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
    feats = output[0][feat_name].split(n_boxes_per_image)
    cur_device = score_list[0].device

    feat_list = []

    for i in range(batch_size):
        dets = output[0]["proposals"][i].bbox / im_scales[i]
        scores = score_list[i]

        max_conf = torch.zeros((scores.shape[0])).to(cur_device)

        for cls_ind in range(1, scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            keep = nms(dets, cls_scores, 0.5)
            max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                         cls_scores[keep],
                                         max_conf[keep])

        keep_boxes = torch.argsort(max_conf, descending=True)[:100]
        feat_list.append(feats[i][keep_boxes])
    return feat_list


def get_detectron_features(image_path, detection_model):
    im, im_scale = image_transform(image_path)
    img_tensor, im_scales = [im], [im_scale]
    current_img_list = to_image_list(img_tensor, size_divisible=32)
    current_img_list = current_img_list.to('cuda')
    with torch.no_grad():
        output = detection_model(current_img_list)
    feat_list = process_feature_extraction(output, im_scales, 'fc6', 0.2)
    return feat_list[0]
