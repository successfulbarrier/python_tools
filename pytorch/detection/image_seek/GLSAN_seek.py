# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   GLSAN_seek.py
# @Time    :   2023/12/30 21:20:22
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   新创建的python文件


import sys
import numpy as np
from tqdm import tqdm
import json
import os
import multiprocessing
import torch
from sklearn.cluster import KMeans
from os import makedirs, listdir
from os.path import join, exists
import cv2
from PIL import Image, ImageDraw, ImageFont
from dataset import my_data_load, xywh_xyxy


def crop_dataset(dataset_path, dst_dataset_dir, cluster_num=4, categories=None, crop_size=300, padding_size=50,
                 normalized_ratio=2):
    dataset = my_data_load(dataset_path, mode="val")
    image_id = 1
    bbox_id = 1

    # create paths
    annotation_dir = join(dst_dataset_dir, 'annotations')
    train_img_dir = join(dst_dataset_dir, 'train')
    if not exists(dst_dataset_dir):
        makedirs(dst_dataset_dir)
        makedirs(train_img_dir)
        makedirs(annotation_dir)

    json_dict = {'images': [], 'type': 'instances', 'categories': [], 'annotations': []}
    cluster_rate = 0
    cluster_images = 0
    for (img, label, img_name) in dataset:
        image = img
        image_shape = image.shape[:2]
        # compute boxes for cropping
        # boxes = instances.gt_boxes.tensor.numpy().astype(np.int32)  # 获取xyxy坐标 [m, 4]
        
        boxes = xywh_xyxy(label[0][0][2:].tolist())
        gt_classes = instances.gt_classes.numpy()   # 获取类别信息 [m]
        points = np.stack(((boxes[:, 0] + boxes[:, 2]) / 2,
                           (boxes[:, 1] + boxes[:, 3]) / 2),
                          axis=1)
        sizes = []
        if len(points) < cluster_num and crop_size >= 0:
            centers = [[] for i in range(len(points))]
            sizes = [[] for i in range(len(points))]
            ranges = [[] for i in range(len(points))]  # [x,y,x,y]
            lbs = np.arange(len(points))
            for i in range(len(points)):
                min_w, min_h, max_w, max_h = boxes[i]
                max_height = max_h - min_h + padding_size
                max_width = max_w - min_w + padding_size
                sizes[i].append(max([max_height, max_width // normalized_ratio, crop_size]))
                sizes[i].append(max([max_height // normalized_ratio, max_width, crop_size]))
                centers[i] = [(min_h + max_h) // 2, (min_w + max_w) // 2]
                ranges[i] = [max(0, centers[i][1] - sizes[i][1] / 2),
                             max(0, centers[i][0] - sizes[i][0] / 2),
                             min(image_shape[1], centers[i][1] + sizes[i][1] / 2),
                             min(image_shape[0], centers[i][0] + sizes[i][0] / 2)]
        else:
            cluster_images += 1
            centers = [[] for i in range(cluster_num)]
            sizes = [[] for i in range(cluster_num)]
            ranges = [[] for i in range(cluster_num)]  # [x,y,x,y]
            kmeans = KMeans(n_clusters=cluster_num)
            classes = kmeans.fit(points)
            lbs = classes.labels_

            for i in range(cluster_num):
                boxes_class_i = boxes[lbs == i]
                boxes_class_i = boxes_class_i.reshape(-1, 2)
                min_w, min_h = boxes_class_i.min(0)
                max_w, max_h = boxes_class_i.max(0)
                max_height = max_h - min_h + padding_size
                max_width = max_w - min_w + padding_size
                sizes[i].append(max([max_height, max_width // normalized_ratio, crop_size]))
                sizes[i].append(max([max_height // normalized_ratio, max_width, crop_size]))
                centers[i] = [(min_h + max_h) // 2, (min_w + max_w) // 2]
                ranges[i] = [max(0, centers[i][1] - sizes[i][1] / 2),
                             max(0, centers[i][0] - sizes[i][0] / 2),
                             min(image_shape[1], centers[i][1] + sizes[i][1] / 2),
                             min(image_shape[0], centers[i][0] + sizes[i][0] / 2)]
        ranges = np.asarray(ranges).astype(np.int32)

        # img = Image.fromarray(image)
        # draw = ImageDraw.Draw(img)
        # for range_i in range(len(ranges)):
        #     r = ranges[range_i]
        #     draw.rectangle(r, outline=(255, 0, 0))
        # img.save('crop.jpg')

        # crop image
        file_name = img_name
        file_head = file_name.split('/')[-1].split('.')[0]
        parent_dir = file_name.split('/')[-2]

        # save original image
        img_name = file_head + '.jpg'
        if parent_dir[0] == 'M':
            if not exists(join(train_img_dir, parent_dir)):
                makedirs(join(train_img_dir, parent_dir))
            img_name = join(parent_dir, img_name)
        ori_file_name = join(train_img_dir, img_name)
        cv2.imwrite(ori_file_name, image)

        image_dict = {'file_name': img_name, 'height': image_shape[0], 'width': image_shape[1],
                      'id': image_id}
        json_dict['images'].append(image_dict)
        ori_boxes = boxes.tolist()
        ori_gt_classes = gt_classes.tolist()

        for obj_i in range(len(ori_boxes)):
            box_i = ori_boxes[obj_i]
            ori_gt_classes_i = ori_gt_classes[obj_i]
            category_id = ori_gt_classes_i + 1  # id index start from 1
            o_width = box_i[2] - box_i[0]
            o_height = box_i[3] - box_i[1]
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': image_id,
                   'bbox': [box_i[0], box_i[1], o_width, o_height], 'category_id': category_id,
                   'id': bbox_id, 'ignore': 0, 'segmentation': []}
            bbox_id += 1
            json_dict['annotations'].append(ann)
        image_id += 1

        # save cropped images
        for range_i in range(len(sizes)):
            # r = transfrom_offsets(centers[i], sizes[i], image_shape[0], image_shape[1])  # [x,y,x,y]
            r = ranges[range_i]
            sub_image = image[r[1]:r[3], r[0]:r[2]]
            sub_image_shape = sub_image.shape[:2]
            sub_img_name = file_head + '_' + str(range_i) + '.jpg'
            if parent_dir[0] == 'M':
                sub_img_name = join(parent_dir, sub_img_name)
            sub_file_name = join(train_img_dir, sub_img_name)
            cv2.imwrite(sub_file_name, sub_image)

            image_dict = {'file_name': sub_img_name, 'height': sub_image_shape[0], 'width': sub_image_shape[1],
                          'id': image_id}
            json_dict['images'].append(image_dict)
            offset = np.tile(r[0:2], 2)
            sub_boxes = (boxes[lbs == range_i] - offset).astype(np.int32).tolist()
            sub_gt_classes = gt_classes[lbs == range_i].tolist()

            for obj_i in range(len(sub_boxes)):
                box_i = sub_boxes[obj_i]
                sub_gt_classes_i = sub_gt_classes[obj_i]
                category_id = sub_gt_classes_i + 1  # id index start from 1
                o_width = box_i[2] - box_i[0]
                o_height = box_i[3] - box_i[1]
                ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': image_id,
                       'bbox': [box_i[0], box_i[1], o_width, o_height], 'category_id': category_id,
                       'id': bbox_id, 'ignore': 0, 'segmentation': []}
                bbox_id += 1
                json_dict['annotations'].append(ann)
            image_id += 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_file = join(annotation_dir, 'train.json')
    # if not exists(json_file):
    #     mknod(json_file)
    json.dump(json_dict, open(json_file, 'w'), indent=4)
    print(cluster_images)
    print(cluster_images/len(dataset))


if __name__ == '__main__':
    crop_dataset("无用", "./")