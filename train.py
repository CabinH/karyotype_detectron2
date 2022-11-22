import os
import cv2
import glob
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

from utils.data import get_dataset_from_coco
from utils.trainer import CocoTrainer
from utils.checkpoint import BestCheckpointer

CLASSES = ['chromosome']

DET2_TRAIN_META = 'maskrcnn_r50_3x_anchor_cs_c1'
DET2_TEST_META  = '{}_test'.format(DET2_TRAIN_META)

image_root      = 'data/cs/images'
annotation_root = 'data/cs/annotations'

# train data
coco_annotations = list(sorted(glob.glob(os.path.join(annotation_root, '*'))))

DatasetCatalog.register(DET2_TRAIN_META, lambda x=image_root, y=coco_annotations[:-1], z=len(CLASSES): get_dataset_from_coco(x, y, z))
DatasetCatalog.register(DET2_TEST_META,  lambda x=image_root, y=coco_annotations[-1:], z=len(CLASSES): get_dataset_from_coco(x, y, z))

MetadataCatalog.get(DET2_TRAIN_META).set(thing_classes=CLASSES)
MetadataCatalog.get(DET2_TEST_META).set(thing_classes=CLASSES)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def config():
    # train configure 
    cfg = get_cfg()
    # model config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = 20
    cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 150000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0]]

    # eval config
    cfg.TEST.EVAL_PERIOD = 1000

    # data config
    cfg.DATASETS.TRAIN = (DET2_TRAIN_META,)
    cfg.DATASETS.TEST  = (DET2_TEST_META, )

    cfg.OUTPUT_DIR = "/data1/huangkaibin/Code/karyotype_detectron2/ckpt/checkpoints_{}".format(DET2_TRAIN_META)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open('config/cfg_{}'.format(DET2_TRAIN_META), 'w') as f:
        f.write(cfg.dump())
    
    return cfg

cfg = config()
# # initiate train
trainer = CocoTrainer(cfg)
trainer.register_hooks([BestCheckpointer()])
trainer.resume_or_load(resume=False)
trainer.train()
