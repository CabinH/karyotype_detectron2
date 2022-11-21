import os
import cv2
import glob
import json
import numpy as np

from detectron2.structures import BoxMode

def refine_rotated_bbox(obb):
    if obb[4] > 90:
        obb[4] = obb[4] - 180
    elif obb[4] < -90:
        obb[4] = obb[4] + 180

    if obb[2] > obb[3]:
        obb[2], obb[3] = obb[3], obb[2]
        obb[4] = obb[4]-90 if obb[4]>0 else obb[4]+90

    return obb

def get_dataset_dicts_for_det2(image_names_list, label_infos_list, mask_infos_list):
    """the functional to get dataset dicts for detectron2"""
    dataset_dicts = []
    for idx, (imagename, labelname, maskname) in enumerate(zip(image_names_list, label_infos_list, mask_infos_list)):
        record = {}
        height, width = cv2.imread(imagename).shape[:2]
        record['file_name'] = imagename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        masks = np.load(maskname)
        labels = np.loadtxt(labelname)
        for mask, label in zip(masks, labels):
            pos  = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            seg = mask2polygon(mask)
            
            if abs(xmax-xmin)<2 or abs(ymax-ymin)<2:
                continue

            obj = {
                'bbox': [xmin, ymin, xmax, ymax],
                'bbox_mode': BoxMode.XYXY_ABS,
                'segmentation': mask2polygon(mask),
                'segmentation': seg,
                'category_id': int(label-1),
            }
            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def get_dataset_from_coco(image_root, coco_annotations, num_class):
    """the functional to get dataset dicts for detectron2"""
    dataset_dicts = []
    for ann_file in coco_annotations:

        with open(ann_file, 'r') as load_f:
            ann_dict = json.load(load_f)

        record = ann_dict['images']
        record['image_id'] = record.pop('id')
        record['file_name'] = os.path.join(image_root, record['file_name'])

        objs = []
        for ann in ann_dict['annotations']:

            obj = {
                'bbox': ann['bbox'],
                'bbox_mode': BoxMode.XYXY_ABS,
                'segmentation': ann['segmentation'],
                'category_id': ann['category_id']-1 if num_class>1 else 0,
            }
            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def get_dataset_from_coco_obb(image_root, coco_annotations, num_class):
    """the functional to get dataset dicts for detectron2"""
    dataset_dicts = []
    for ann_file in coco_annotations:
        with open(ann_file, 'r') as load_f:
            ann_dict = json.load(load_f)

        record = ann_dict['images']
        record['image_id'] = record.pop('id')
        record['file_name'] = os.path.join(image_root, record['file_name'])

        objs = []
        for ann in ann_dict['annotations']:
            obj = {
                'bbox': refine_rotated_bbox(ann['bbox']),
                'bbox_mode': BoxMode.XYWHA_ABS,
                'segmentation': ann['segmentation'],
                'category_id': ann['category_id']-1 if num_class>1 else 0,
            }
            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# without test...
def convert_to_coco_dict(dataset_dicts, class_name_list):

    coco_images = []
    coco_annotations = []
    coco_categories = [{"id": index, "name": value, "supercategory": "empty"} for
                  index, value in enumerate(class_name_list)]
    for image_dict in dataset_dicts:
        coco_image = {
                'id': int(image_dict['image_id']),
                'width': image_dict['width'],
                'height': image_dict['height'],
                'file_name': image_dict['file_name'],
        }
        coco_images.append(coco_image)

        for annotation in image_dict['annotations']:
            coco_annotation = {}

            # COCO requirement: XYWH box format
            bbox = annotation["bbox"]
            bbox_mode = annotation["bbox_mode"]
            # Computing areas using bounding boxes
            bbox_xy = BoxMode.convert(bbox, bbox_mode, BoxMode.XYXY_ABS)
            area = Boxes([bbox_xy]).area()[0].item()

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = float(area)
            coco_annotation["iscrowd"] = 0
            coco_annotation["category_id"] = annotation["category_id"]

            coco_annotations.append(coco_annotation)

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }

    coco_dict = {
        "info": info,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
        "licenses": None,
    }
    return coco_dict

def convert_to_coco_json(output_file, dataset_dicts, class_name_list):
    coco_dict = convert_to_coco_dict(dataset_dicts, class_name_list)

    PathManager.mkdirs(os.path.dirname(output_file))
    with PathManager.open(output_file, "w") as f:
        json.dump(coco_dict, f)

def mask2polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []

    contours = sorted(list(contours), key=lambda x:len(x), reverse=True)

    for object in contours[:1]:
        coords = []

        for point in object:
            coords.append(int(point[0][0]))
            coords.append(int(point[0][1]))

        polygons.append(coords)
    return polygons

def mask2rle(img):
    '''
    Efficient implementation of mask2rle, from @paulorzp
    --
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    Source: https://www.kaggle.com/xhlulu/efficient-mask2rle
    '''
    pixels = img.T.flatten()
    pixels = np.pad(pixels, ((1, 1), ))
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    Source: https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

if __name__ == '__main__':
    
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.utils.visualizer import Visualizer, ColorMode

    CLASSES_TYPE = {'c1': ['chromosome'],
                    'c24': ['1',  '2',  '3',  '4',  '5',  '6',
                            '7',  '8',  '9',  '10', '11', '12',
                            '13', '14', '15', '16', '17', '18',
                            '19', '20', '21', '22', 'X',  'Y',]
                    }
    CLASSES = CLASSES_TYPE['c24']

    image_root = '../data/syn/images'
    annotation_root = '../data/syn/annotations'
    obb_annotations = list(sorted(glob.glob(os.path.join(annotation_root, '*'))))[-10:]

    DatasetCatalog.register('vis', lambda x=image_root, y=obb_annotations: get_dataset_from_coco_obb(x, y))
    MetadataCatalog.get('vis').set(thing_classes=CLASSES)
    metadata = MetadataCatalog.get('vis')
    
    # data_dicts = get_dataset_from_coco_obb(image_root, obb_annotations)
    data_dicts = get_dataset_from_coco(image_root, obb_annotations, len(CLASSES))
    for i, data_dict in enumerate(data_dicts):
        image = cv2.imread(data_dict['file_name'])
        visualizer = Visualizer(image, metadata=metadata)
        out = visualizer.draw_dataset_dict(data_dict)
        cv2.imwrite(f'{i}.png', out.get_image())
