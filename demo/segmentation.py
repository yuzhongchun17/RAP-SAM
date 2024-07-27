# File: segmentation_inference.py
import cv2
import numpy as np
import ast
from mmengine.logging import print_log
from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes

def init_inferencer(model_config, weights, device='cuda:0', palette='none'):
    inferencer = DetInferencer(model=model_config, weights=weights, device=device, palette=palette)
    return inferencer

def infer(inferencer, img, texts=None, pred_score_thr=0.6, no_save_vis=False, no_save_pred=True, print_result=False, custom_entities=False):
    # if texts and texts.startswith('$:'):
    #     dataset_name = texts[3:].strip()
    #     class_names = get_classes(dataset_name)
    #     texts = [tuple(class_names)]
    # print(texts)

    # if custom_entities:
    #     class_names = texts.split(' . ')
    #     texts = [tuple(class_names)]

    # img = cv2.imread(image_path)
    
    call_args = {
        'inputs': img, # can be img or image path
        # 'out_dir': output_path,  
        'pred_score_thr': pred_score_thr,
        'show': False,  # Don't show popup
        'no_save_vis': no_save_vis,
        'no_save_pred': no_save_pred,
        'print_result': print_result,
        'texts': texts
    }

    results = inferencer(**call_args)
    if print_result:
        print_log(f'Detection results: {results}')
    # visualization_image = results['visualization'][0]
    # visualization_image = cv2.cvtColor(visualization_image, cv2.COLOR_RGB2BGR) # BGR output
    # cv2.imshow("Segmentation Visualization", visualization_image)
    # print("Press 'q' to close the window.")
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()

    # print(visualization_image.shape)  # Expect something like (height, width, channels)
    # print(visualization_image.dtype)  # Typically should be np.uint8

    return results

import numpy as np
from pycocotools import mask as coco_mask

def get_combined_binary_mask(inference, target_label=0, score_threshold=0.7):
    # Extract information
    labels = np.array(inference['predictions'][0]['labels'])
    scores = np.array(inference['predictions'][0]['scores'])
    masks = inference['predictions'][0]['masks']

    # Get indices where both conditions are met
    indices = np.where((labels == target_label) & (scores >= score_threshold))[0]

    # Extract masks that meet the conditions
    mask_rle = [masks[i] for i in indices]

    # Decode masks from RLE to binary
    masks_binary = [coco_mask.decode(m) for m in mask_rle]

    # Combine all masks
    if masks_binary:
        mask_binary_combined = np.maximum.reduce(masks_binary)
        return mask_binary_combined
    else:
        return None

