import segmentation
import cv2

def test_segmentation():
    # Path to your model config and checkpoint
    config_path = 'configs/rap_sam/eval_rap_sam_coco.py'
    checkpoint_path = 'rapsam_r50_12e.pth'
    image_path = 'demo/demo_person.jpg'
    output_path = 'output/vis'
    
    # Initialize the model
    model = segmentation.init_inferencer(config_path, checkpoint_path)
    
    # Perform segmentation (result is output RGB)
    inference = segmentation.infer(model, image_path)

    # ----------------------------------
    # if inference:
    #     print(f"Type of results: {type(inference)}")
    #     print("Keys in results:", inference.keys())
        
    #     # Print the type of the values for each key
    #     for key in inference.keys():
    #         print(f"Type of values for {key}: {type(inference[key])}")
    #         if isinstance(inference[key], list):
    #             print(f"Length of list for {key}: {len(inference[key])}")
    #             if inference[key]:
    #                 print(f"Type of elements in {key}: {type(inference[key][0])}")
    #         elif isinstance(inference[key], dict):
    #             print(f"Keys in the dictionary for {key}: {inference[key].keys()}")
    # else:
    #     print("No results were returned.")  
    # ----------------------------------

    # print(type(inference['predictions'][0])) 
    # print(inference['predictions'][0].keys())
    # print(len(inference['predictions'][0]['labels']))
    # print(inference['predictions'][0]['labels'])
    # print(len(inference['predictions'][0]['scores']))
    # print(inference['predictions'][0]['scores'])
    # print(len(inference['predictions'][0]['masks']))

    import numpy as np
    from pycocotools import mask as coco_mask
    # Extract info from inference
    labels = np.array(inference['predictions'][0]['labels']) # list to np array
    scores = np.array(inference['predictions'][0]['scores'])
    masks = inference['predictions'][0]['masks'] # Fetch masks

    # Get indices (where both conditions are met)
    threshold = 0.7
    indices = np.where((labels == 0) & (scores >= threshold))[0]

    # Extract the masks that meet the conditions
    selected_masks = [masks[i] for i in indices]

    # Decoding: coco rle --> binary mask
    decoded_masks = [coco_mask.decode(m) for m in selected_masks]

    # Combine all masks 
    if decoded_masks:
        combined_mask = np.maximum.reduce(decoded_masks)
    else:
        # If no masks are decoded, create an empty mask 
        height, width = selected_masks[0]['size']
        combined_mask = np.zeros((height, width), dtype=np.uint8) 

    # binary_mask = coco_mask.decode(selected_masks)
    cv2.imshow('binary_mask',combined_mask *150)
    # print("Press 'q' to close the window.")
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()
    
    # print(inference['predictions'][0]['panoptic_seg'])
    # import numpy as np

    # Assuming 'panoptic_seg' is your array
    # panoptic_seg = inference['predictions'][0]['panoptic_seg']
    # print(np.nonzero(panoptic_seg[:,:,2]))

    # panoptic_seg = inference['predictions'][0]['panoptic_seg']
    # grayscale = cv2.cvtColor(panoptic_seg, cv2.COLOR_RGB2GRAY)
    # print(grayscale.shape)
    # cv2.imshow('seg',panoptic_seg)
    # print("Press 'q' to close the window.")
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()


    # # Initialize a list to keep track of channels with non-zero values
    # non_zero_channels = []

    # # Check each channel
    # for i in range(panoptic_seg.shape[2]):
    #     if np.any(panoptic_seg[:, :, i] != 0):  # Check if there are any non-zero values in this channel
    #         non_zero_channels.append(i)

    # print("Channels with non-zero values:", non_zero_channels)


    # cv2.imshow("Panoptic Segmentation Prediction", inference['predictions'][0]['panoptic_seg'])
    # # cv2.imshow("Segmentation Visualization", inference['visualization'][0])
    # print("Press 'q' to close the window.")
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()

    # if results:
    #     print(f"Type of results: {type(results)}")
    #     print("Keys in results:", results.keys())
        
    #     # Print the type of the values for each key
    #     for key in results.keys():
    #         print(f"Type of values for {key}: {type(results[key])}")
    #         if isinstance(results[key], list):
    #             print(f"Length of list for {key}: {len(results[key])}")
    #             if results[key]:
    #                 print(f"Type of elements in {key}: {type(results[key][0])}")
    #         elif isinstance(results[key], dict):
    #             print(f"Keys in the dictionary for {key}: {results[key].keys()}")
    # else:
    #     print("No results were returned.")
    # Output results
    # print("Segmentation Results:", results)
    # Depending on your setup, you might want to visualize the results or check specific output details
    # For example, drawing the results on the image, which isn't covered here since we're focusing on a simple test.

if __name__ == '__main__':
    test_segmentation()