import numpy as np
import torch
import random
import os
import copy
import platform
import json
import re
from lib.datasets.dataset_util import *
from lib.utils import data_utils
from pdb import set_trace
import torch.utils.data as data
from tqdm import tqdm
import cv2
from lib.utils.snake import snake_voc_utils, snake_config, visualize_utils
import math


def make_dataset(mode, split=0, min_fg=100,data_root=None, sample_list=None,scan_ids=None ,sub_list=None, sub_val_list=None):
    assert split in [0, 1, 2, 3, 4]

    data_list = []
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}

    # Setting 2: images contains the test-class should remove from the train dataset

    if mode == 'train':
        for sub_c in sub_list:
            sub_class_file_list[sub_c] = []
        for slc in tqdm(range(len(sample_list))):
            line = sample_list[slc].replace('\n', '')
            if int(re.findall(r'\d+', line)[0]) not in scan_ids:
                continue
            line = os.path.join(data_root, line)
            data = np.load(line)
            image, label = data['data'], data['label']
            label_class = np.unique(label).tolist()

            # if set(sub_val_list) in set(label_class):
            #     continue

            if 0 in label_class:
                label_class.remove(0)

            new_label_class = []

            for c in label_class:
                if c in sub_list:
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    tmp_label[target_pix[0], target_pix[1]] = 1
                    if tmp_label.sum() >= int(min_fg):
                        new_label_class.append(c)

            label_class = new_label_class

            if len(label_class) > 0:
                data_list.append(line)
                for c in label_class:
                    if c in sub_list:
                        sub_class_file_list[c].append(line)

    elif mode == 'val' or mode =='test':
        for sub_c in sub_val_list:
            sub_class_file_list[sub_c] = []
        for slc in tqdm(range(len(sample_list))):
            line = sample_list[slc].replace('\n', '')
            if int(re.findall(r'\d+', line)[0]) not in scan_ids:
                continue
            line = os.path.join(data_root, line)
            data = np.load(line)
            image, label = data['data'], data['label']
            label_class = np.unique(label).tolist()

            if 0 in label_class:
                label_class.remove(0)

            new_label_class = []

            for c in label_class:
                if c in sub_val_list:
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    tmp_label[target_pix[0], target_pix[1]] = 1
                    if tmp_label.sum() >= int(min_fg):
                        new_label_class.append(c)

            label_class = new_label_class

            if len(label_class) > 0:
                data_list.append(line)
                for c in label_class:
                    if c in sub_val_list:
                        sub_class_file_list[c].append(line)

    print("Checking image&label pair {} list done! ".format(split))
    return data_list, sub_class_file_list



class Dataset(data.Dataset):
    def __init__(self, which_dataset, data_dir, list_dir,idx_split, mode, transforms=None, min_fg = ' ',shot = 1, test_class=None, tile_z_dim=3, **kwargs):

        assert mode in ['train', 'val', 'test']
        self.img_modality = DATASET_INFO[which_dataset]['MODALITY']
        self.sep = DATASET_INFO[which_dataset]['_SEP']
        self.real_label_name = DATASET_INFO[which_dataset]['REAL_LABEL_NAME']

        self.transforms = transforms
        self.mode = mode
        self.tile_z_dim = tile_z_dim

        #find split in the data folder
        self.shot = shot
        self.base_dir = data_dir
        self.list_dir = list_dir
        self.sample_list = open((list_dir)).readlines()
        self.img_pids = np.unique([ int(re.findall(r'\d+', sample)[0]) for sample in self.sample_list]).tolist()


        # experiment configs
        # test_class means choose a class to be segmented and the rest are used for traing
        if which_dataset == 'SABS':
            if test_class == 'SPLEEN':
                self.sub_list = [2, 3, 6]
                self.sub_val_list = [1]
            elif test_class == 'KID_R':
                self.sub_list = [1, 3, 6]
                self.sub_val_list = [2]
            elif test_class == 'KID_L':
                self.sub_list = [1, 2, 6]
                self.sub_val_list = [3]
            elif test_class == 'LIVER':
                self.sub_list = [1, 2, 3]
                self.sub_val_list = [6]
        elif which_dataset == 'CHAOST2':
            if test_class == 'SPLEEN':
                self.sub_list = [1, 2, 3]
                self.sub_val_list = [4]
            elif test_class == 'KID_R':
                self.sub_list = [1, 3, 4]
                self.sub_val_list = [2]
            elif test_class == 'KID_L':
                self.sub_list = [1, 2, 4]
                self.sub_val_list = [3]
            elif test_class == 'LIVER':
                self.sub_val_list = [2, 3, 4]
                self.sub_val_list = [1]

        self.idx_split = idx_split
        self.scan_ids = self.get_scanids(mode, idx_split)
        self.min_fg = min_fg if isinstance(min_fg, str) else str(min_fg)

        if self.mode == 'train':
            self.data_list, self.sub_class_file_list = make_dataset(mode=self.mode, split=self.idx_split,
                                                                    min_fg=self.min_fg, data_root=self.base_dir,
                                                                    sample_list=self.sample_list,
                                                                    scan_ids=self.scan_ids, sub_list=self.sub_list,
                                                                    sub_val_list=self.sub_val_list)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_list)

        elif self.mode == 'val' or 'test':
            self.data_list, self.sub_class_file_list = make_dataset(mode=self.mode, split=self.idx_split,
                                                                    min_fg=self.min_fg, data_root=self.base_dir,
                                                                    sample_list=self.sample_list,
                                                                    scan_ids=self.scan_ids, sub_list=self.sub_list,
                                                                    sub_val_list=self.sub_val_list)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list)

        self.transforms = transforms
        print("###### Dataset Initial Finished!!: ######")

    def get_scanids(self, mode, idx_split):
        """
        Load scans by train-test split
        leaving one additional scan as the support scan. if the last fold, taking scan 0 as the additional one
        Args:

            idx_split: index for spliting cross-validation folds

        Returns:

            val_ids: index for validation folds
        """
        val_ids = copy.deepcopy(self.img_pids[self.sep[idx_split]: self.sep[idx_split + 1]])
        if mode == 'train':
            return [ii for ii in self.img_pids if ii not in val_ids]
        elif mode == 'val':
            return val_ids


    def __len__(self):
        """
         copy-paste from basic naive dataset configuration
        Returns:

        """
        return len(self.data_list)


    def process_info(self, index):

        data_path = self.data_list[index]

        data = np.load(data_path)
        image, label = data['data'], data['label']

        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)

        new_label_class = []
        for c in label_class:
            if c in self.sub_val_list:
                if self.mode == 'val' or self.mode == 'test':
                    # setting 2 : removing any images contains test a testing class
                    new_label_class.append(c)
            if c in self.sub_list:
                if self.mode == 'train':
                    new_label_class.append(c)
        label_class = new_label_class
        assert len(label_class) > 0


        class_chosen = label_class[random.randint(1, len(label_class)) - 1]
        class_chosen =  class_chosen
        target_pix = np.where(label == class_chosen)
        label[:,:] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0], target_pix[1]] = 1

        if np.sum(label) <=  100:
            return self.__getitem__(torch.randint(low=0, high=self.__len__() - 1, size=(1,)))

        label = label.astype('uint8')
        raw_label = label.copy()
        query_coutours, hierarchy = cv2.findContours(raw_label, mode=cv2.RETR_EXTERNAL, method=1)
        query_coutour = query_coutours[0]
        query_coutour = query_coutour.squeeze(1)

        image = torch.from_numpy(np.squeeze(image, 2))
        label = torch.from_numpy(np.squeeze(label, 2))

        if self.tile_z_dim:
            image = image.repeat( [3, 1, 1])
            assert image.ndimension() == 3, f'actual dim {image.ndimension()}'

        file_class_choen = self.sub_class_file_list[class_chosen]
        num_file = len(file_class_choen)

        support_data_path_list = []
        support_idx_list = []

        for k in range(self.shot):
            support_idx = random.randint(1, num_file) - 1
            support_data_path =  data_path
            while ((support_data_path == data_path) or (support_idx in support_idx_list)):
                support_idx = random.randint(1, num_file) -1
                support_data_path = file_class_choen[support_idx]
            support_idx_list.append(support_idx)
            support_data_path_list.append(support_data_path)

        output = {}

        support_image_list = []
        support_label_list = []
        subcls_list = []
        support_coutour_list = []
        cls_id_list = []
        cls_id_list.append(class_chosen)

        for k in range(self.shot):
            if self.mode == 'train':
                subcls_list.append(self.sub_list.index(class_chosen))
            else:
                subcls_list.append(self.sub_val_list.index(class_chosen))
            support_data = np.load(support_data_path_list[k])
            support_image, support_label = support_data['data'], support_data['label']
            support_image = np.squeeze(support_image, 2)
            support_label = np.squeeze(support_label, 2)
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:, :] = 0
            support_label[target_pix[0], target_pix[1]] = 1
            support_label = support_label.astype('uint8')
            raw_support_label = support_label.copy()
            support_coutours, hierarchy = cv2.findContours(raw_support_label, mode=cv2.RETR_EXTERNAL, method=1)
            tmp_countour_list = []

            for coutour in support_coutours:
                coutour = coutour.squeeze(1)
                tmp_countour_list.append(coutour)
            support_coutour_list.append(tmp_countour_list)

            if support_image.shape[0] != support_label.shape[0] or  support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch:" + support_data_path_list + "\n"))

            support_image = torch.from_numpy(support_image)
            support_label = torch.from_numpy(support_label)

            if self.tile_z_dim:
                support_image = support_image.repeat( [3,1,1] )
                assert support_image.ndimension() == 3, f'actual dim {support_image.ndimension()}'

            support_image_list.append(support_image)
            support_label_list.append(support_label)


        assert len(support_image_list) == self.shot or len(support_label_list) == self.shot
        if self.transforms is not None:
            image, label = self.transforms(image, label)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transforms(support_image_list[k], support_label_list[k])

        output.update({'inp':image, 'query_label':label, 'query_contour':[query_coutour]})
        output.update({'support_image_list':support_image_list, 'support_label_list':support_label_list,'support_countour_list':support_coutour_list})
        output.update({'cls_id':class_chosen})

        return output

    def augment(self, img, mode, instance_polys):
        height, width = img.shape[0], img.shape[1]
        center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        scale = max (img.shape[0], img.shape[1])
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

    def prepare_detection(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):

        # 返回detect_box 的center大小和box的长宽
        # 输入的box是在x轴和y轴上输出最大最小的得到的，这里输出的box是在center上加上w/2和h/2得到的

        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct = np.round(ct).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        return decode_box

    def  prepare_init(self, box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys):
        # 该函数返回菱形的40个采样点
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        img_init_poly = snake_voc_utils.get_init(box) # 获得一个菱形的边框（4个点）
        img_init_poly = snake_voc_utils.uniformsample(img_init_poly, snake_config.init_poly_num)
        can_init_poly = snake_voc_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
        img_gt_poly = extreme_point
        can_gt_poly = snake_voc_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        i_it_4pys.append(img_init_poly)
        c_it_4pys.append(can_init_poly)
        i_gt_4pys.append(img_gt_poly)
        c_gt_4pys.append(can_gt_poly)


    def prepare_evolution(self, poly, extreme_point, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys):
        # 返回八边形的128个点
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        octagon = snake_voc_utils.get_octagon(extreme_point)
        img_init_poly = snake_voc_utils.uniformsample(octagon, snake_config.poly_num)
        can_init_poly = snake_voc_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)

        img_gt_poly = snake_voc_utils.uniformsample(poly, len(poly) * snake_config.gt_poly_num)
        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        can_gt_poly = snake_voc_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = snake_voc_utils.get_extreme_points(instance)
            extreme_points.append(points)
        return extreme_points


    def __getitem__(self, index):

        index = index % len(self.data_list)
        output = self.process_info(index)
        image, instance_polys, cls_id = output['inp'], output['query_contour'], output['cls_id']

        extreme_points = self.get_extreme_points(instance_polys)

        ct_hm = np.zeros([14, 128, 128], dtype=np.float32)

        wh = []
        ct_cls = []
        ct_ind = []

        # init
        # i_it_4pys 在diamond上的40个采样点
        # c_it_4pys 标准化后的40个采样点
        # i_gt_4pys groundtruth的4个点
        # c_gt_4pys 标准化的groundtruth4个点

        i_it_4pys = []
        c_it_4pys = []
        i_gt_4pys = []
        c_gt_4pys = []

        # evolution
        # i_it_pys 在八边形上采样的128个点
        # c_it_pys 标准化后的128个点
        # i_gt_py 在gt上采样的128个点
        # c_gt_py 在gt上标准化后的128个点

        i_it_pys = []
        c_it_pys = []
        i_gt_pys = []
        c_gt_pys = []

        poly = instance_polys[0]
        extreme_point = extreme_points[0]

        x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
        x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
        bbox = [x_min, y_min, x_max, y_max]
        h, w = y_max - y_min + 1, x_max - x_min + 1

        if h <= 1 or w <= 1:
            return self.__getitem__(index+1)

        self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
        self.prepare_init(bbox, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys)
        self.prepare_evolution(poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)

        detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
        evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}

        output.update(detection)
        output.update(init)
        output.update(evolution)

        ct_num = len(ct_ind)
        meta = {'ct_num': ct_num}
        output.update({'meta':meta})

        return output
















































