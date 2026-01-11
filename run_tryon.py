"""
Virtual Try-On Pipeline
Author: Sufi Shaikh
Description: End-to-end virtual try-on pipeline using OpenPose,
Graphonomy, Detectron2, and HR-VITON.
"""
import os
import shutil
import warnings
import numpy as np
import cv2
import glob
import argparse
import draw_agnostic

warnings.filterwarnings("ignore", category=FutureWarning)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--background', type=bool, default=True, help='Define removing background or not')
    opt = parser.parse_args()

    img=cv2.imread("./input/model.jpg")

    # (768, 1024) resize
    if img.shape[:2] != (1024, 768):
        model_img = cv2.resize(img, (768, 1024))
        logging.info("Input image resized to (768, 1024)")

    else:
        model_img = img
        logging.info("Input image already in required resolution (768, 1024)")


    # save
    cv2.imwrite("./model.jpg", model_img)

    img=cv2.imread("model.jpg")
    img=cv2.resize(img,(384,512))
    cv2.imwrite('resized_img.jpg',img)
    
    # Get mask of cloth
    logging.info("Generating cloth segmentation mask")
    os.system("python clothseg.py")

    # OpenPose Keypoints JSON
    input_dir = "/content/Fashion-U-Want-Virtual-Try-On/input"
    input_image = "model.jpg"
    output_json_path = "/content/Fashion-U-Want-Virtual-Try-On/HR-VITON/test/test/openpose_json"
    json_filename = "00001_00_keypoints.json"
    os.makedirs(output_json_path, exist_ok=True)

    logging.info("Running OpenPose for human pose estimation")
    os.system(f"cd /content/Fashion-U-Want-Virtual-Try-On/openpose && ./build/examples/openpose/openpose.bin "
            f"--image_dir {input_dir} "
            f"--write_json {output_json_path} "
            f"--model_folder ./models/ "
            f"--render_pose 0 "
            f"--display 0")

    generated_json = os.path.join(output_json_path, f"{os.path.splitext(input_image)[0]}_keypoints.json")
    target_json = os.path.join(output_json_path, json_filename)

    if os.path.exists(generated_json):
        shutil.move(generated_json, target_json)
        logging.info(f"OpenPose keypoints saved as {target_json}")
    else:
        logging.error("OpenPose keypoints JSON file was not generated")
    os.chdir("../")

    # Graphonomy 
    logging.info("Generating human semantic segmentation using Graphonomy")
    os.chdir("/content/Fashion-U-Want-Virtual-Try-On/Graphonomy-master")
    os.system("python exp/inference/inference.py --loadmodel ./inference.pth --img_path ../resized_img.jpg --output_path ../ --output_name /resized_segmentation_img")
    os.chdir("../")

    output_dir = "./HR-VITON/test/test/image"
    os.makedirs(output_dir, exist_ok=True)
    
    
    logging.info("User input required: Please draw the agnostic mask using the mouse")
    draw_agnostic.draw_agnostic_mask("model.jpg", "HR-VITON/test/test/agnostic-v3.2/custom_agnostic_mask.png")

    # HR-VITON
    logging.info("Running HR-VITON to generate final try-on image")
    os.chdir("./HR-VITON")
    os.system("python3 test_generator.py --cuda True --test_name test1 --tocg_checkpoint mtviton.pth --gpu_ids 0 --gen_checkpoint gen.pth --datasetting unpaired --data_list t2.txt --dataroot ./test") 

    
    l=glob.glob("./Output/*.png")

    mask_img=cv2.imread("HR-VITON/test/test/agnostic-v3.2/custom_agnostic_mask.png", cv2.IMREAD_GRAYSCALE)
    back_ground = cv2.imread("./model.jpg") 
    if opt.background:
        for i in l:
            img=cv2.imread(i)
            img=cv2.bitwise_and(img, img, mask=mask_img)
            img=img+back_ground
            cv2.imwrite(i, img)
    else:
        for i in l:
            img=cv2.imread(i)
            cv2.imwrite(i, img)

    logging.info("Virtual try-on pipeline completed successfully")
    os.chdir("../")
    cv2.imwrite("./input/finalimg.png", img)

    if __name__ == "__main__":
    main()
