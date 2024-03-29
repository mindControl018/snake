from lib.config import cfg


class DatasetCatalog(object):
    dataset_attrs = {
        'CocoTrain': {
            'id': 'coco',
            'data_root': '/chengang1/LiuYan/Dataset/COCO/train2017',
            'ann_file': '/chengang1/LiuYan/Dataset/COCO/annotations/instances_train2017.json',
            'split': 'train'
        },
        'CocoVal': {
            'id': 'coco',
            'data_root': '/chengang1/LiuYan/Dataset/COCO/val2017',
            'ann_file': '/chengang1/LiuYan/Dataset/COCO/annotations/instances_val2017.json',
            'split': 'test'
        },
        'CocoMini': {
            'id': 'coco',
            'data_root': '/chengang1/LiuYan/Dataset/COCO/val2017',
            'ann_file': '/chengang1/LiuYan/Dataset/COCO/annotations/instances_val2017.json',
            'split': 'mini'
        },
        'CocoTest': {
            'id': 'coco_test',
            'data_root': 'data/coco/test2017',
            'ann_file': 'data/coco/annotations/image_info_test-dev2017.json',
            'split': 'test'
        },
        'CityscapesTrain': {
             'data_root': 'data/cityscapes/leftImg8bit',
            'ann_file': ('data/cityscapes/annotations/train', 'data/cityscapes/annotations/train_val'),
            'split': 'train'
        },
        'CityscapesVal': {
            'id': 'cityscapes',
            'data_root': 'data/cityscapes/leftImg8bit',
            'ann_file': 'data/cityscapes/annotations/val',
            'split': 'val'
        },
        'CityscapesCocoVal': {
            'id': 'cityscapes_coco',
            'data_root': 'data/cityscapes/leftImg8bit/val',
            'ann_file': 'data/cityscapes/coco_ann/instance_val.json',
            'split': 'val'
        },
        'CityCocoBox': {
            'id': 'cityscapes_coco',
            'data_root': 'data/cityscapes/leftImg8bit/val',
            'ann_file': 'data/cityscapes/coco_ann/instance_box_val.json',
            'split': 'val'
        },
        'CityscapesMini': {
            'id': 'cityscapes',
            'data_root': 'data/cityscapes/leftImg8bit',
            'ann_file': 'data/cityscapes/annotations/val',
            'split': 'mini'
        },
        'CityscapesTest': {
            'id': 'cityscapes_test',
            'data_root': 'data/cityscapes/leftImg8bit/test'
        },
        'SbdTrain': {
            'id': 'sbd',
            'data_root': '/home/chengang/chengang/LiuYan_1/LiuYan/Dataset/sbd/img',
            'ann_file': '/home/chengang/chengang/LiuYan_1/LiuYan/Dataset/sbd/annotations/sbd_trainval_box_instance.json',
            'split': 'train'
        },
        'SbdVal': {
            'id': 'sbd',
            'data_root': '/home/chengang/chengang/LiuYan_1/LiuYan/Dataset/sbd/img',
            'ann_file': '/home/chengang/chengang/LiuYan_1/LiuYan/Dataset/sbd/annotations/sbd_trainval_instance.json',
            'split': 'val'
        },
        'SbdMini': {
            'id': 'sbd',
            'data_root': '/home/chengang/chengang/LiuYan_1/LiuYan/Dataset/sbd/img',
            'ann_file': '/home/chengang/chengang/LiuYan_1/LiuYan/Dataset/sbd/annotations/sbd_trainval_instance.json',
            'split': 'mini'
        },
        'VocVal': {
            'id': 'voc',
            'data_root': 'data/voc/JPEGImages',
            'ann_file': 'data/voc/annotations/voc_val_instance.json',
            'split': 'val'
        },
        'KinsTrain': {
            'id': 'kins',
            'data_root': 'data/kitti/training/image_2',
            'ann_file': 'data/kitti/training/instances_train.json',
            'split': 'train'
        },
        'KinsVal': {
            'id': 'kins',
            'data_root': 'data/kitti/testing/image_2',
            'ann_file': 'data/kitti/testing/instances_val.json',
            'split': 'val'
        },
        'KinsMini': {
            'id': 'kins',
            'data_root': 'data/kitti/testing/image_2',
            'ann_file': 'data/kitti/testing/instances_val.json',
            'split': 'mini'
        },
        'SABSTrain':{
            'id':'SABS',
            'mode': 'train',
            'which_dataset': 'SABS',
            'idx_split': 0,
            'min_fg': '100',
            'shot': 1,
            'test_class': 'SPLEEN',
            'tile_z_dim': 3,
            'data_dir': '/home/chengang/chengang/LiuYan_1/LiuYan/SSL-ALPNet/data/SABS/case_slice_data',
            'list_dir': '/home/chengang/chengang/LiuYan_1/LiuYan/SSL-ALPNet/data/SABS/slice_list.txt',
        },
        'SABSVal':{
            'id':'SABS',
            'mode': 'val',
            'which_dataset': 'SABS',
            'idx_split': 0,
            'min_fg': '100',
            'shot': 1,
            'test_class': 'SPLEEN',
            'tile_z_dim': 3,
            'data_dir': '/home/chengang/chengang/LiuYan_1/LiuYan/SSL-ALPNet/data/SABS/case_slice_data',
            'list_dir': '/home/chengang/chengang/LiuYan_1/LiuYan/SSL-ALPNet/data/SABS/slice_list.txt',
        }
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()

