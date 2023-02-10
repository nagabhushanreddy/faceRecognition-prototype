import argparse
import os
import shutil

from utils.logger import logger

import cv2
import face_recognition
import numpy as np


def extractFacePositions(all_face_locations: list):
    facePositions = []
    for index, current_face_location in enumerate(all_face_locations):
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        facePositions.append(
            {"top_pos": top_pos, "right_pos": right_pos, "bottom_pos": bottom_pos, "left_pos": left_pos})
    return facePositions


def writeOutput(output, image, face_number, face):
    os.makedirs(output, exist_ok=True)
    image_file_name = os.path.basename(image)
    file_name_as_dir, extension = os.path.splitext(image_file_name)
    os.makedirs(f'{output}/{file_name_as_dir}', exist_ok=True)
    image_file_path = f'{output}/{file_name_as_dir}/{image_file_name}'
    face_file_path = f'{output}/{file_name_as_dir}/{file_name_as_dir}_face_{face_number}{extension}'
    shutil.copy(image, image_file_path)
    cv2.imwrite(face_file_path, face)
    return {'image_file_path': image_file_path, 'face_file_path': face_file_path}


def faceDetection(args):
    logger.info(f'Arguments passed are, {args}')
    image_to_detect = cv2.imread(args.get('image'))
    display_images = []
    # Display image
    # cv2.imshow('Input Image',image_to_detect)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()
    display_images.append(image_to_detect)
    all_face_locations = face_recognition.face_locations(image_to_detect, model=args.get('detection-method'))
    logger.info(f'There are {len(all_face_locations)} faces in image {args.get("image")}')
    facePositions = extractFacePositions(all_face_locations)
    for index, facePosition in enumerate(facePositions):
        logger.info(f'Display {index + 1} face')
        current_face = image_to_detect[facePosition.get('top_pos'):facePosition.get('bottom_pos'),
                       facePosition.get('left_pos'):facePosition.get('right_pos')]
        display_images.append(current_face)
        current_face = cv2.resize(current_face, (100, 100), interpolation=cv2.INTER_AREA)
        border_face_image_to_detect = cv2.rectangle(image_to_detect.copy(),
                                               (facePosition.get('left_pos'), facePosition.get('top_pos')),
                                               (facePosition.get('right_pos'), facePosition.get('bottom_pos')),
                                               color=(50, 150, 255), thickness=8
                                               )
        display_images.append(border_face_image_to_detect)

        if args.get('output'):
            writeOutput(args.get('output'), args.get('image'), (index + 1), current_face)

    rows = max([img.shape[0] for img in display_images])
    cols = sum([img.shape[1] for img in display_images])
    # Create a black canvas to hold the images
    canvas = np.zeros((rows, cols, 3), dtype=np.uint8)

    # Copy each image onto the canvas
    x_offset = 0
    for img in display_images:
        h, w = img.shape[:2]
        canvas[:h, x_offset:x_offset + w] = img
        x_offset += w

    cv2.imshow('Face Detection Canvas', canvas)
    cv2.waitKey(20000)
    cv2.destroyAllWindows()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to directory of images for detecting")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-o", "--output", required=False,
                help="path to output directory to save output files")

# Main method starts here.
if __name__ == '__main__':
    args = vars(ap.parse_args())
    exit(faceDetection(args))
