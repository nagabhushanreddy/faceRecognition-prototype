import argparse

import cv2
import face_recognition
import numpy as np

from image_face_alignment import face_alignment
from image_face_detection import extractFacePositions
from utils.logger import logger
from utils.text_to_image import text_to_image

def extract_feature_vector(face_image_path, detection_method = 'cnn'):
    feature_vector = None
    image = cv2.imread(face_image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    all_face_locations = face_recognition.face_locations(rgb, model= detection_method)
    for index, face_location in enumerate(all_face_locations):
        feature_vector = face_recognition.face_encodings(rgb, [face_location])
    return feature_vector

def encode_face(image,output):
    output_image_paths = face_alignment(image, output)
    for index,face_image_path in enumerate(output_image_paths):
        face_file_path = face_image_path.get('face_file_path')
        face_vector = extract_feature_vector(face_file_path)
        face_image_path['vector'] = face_vector.pop().tolist()
        logger.info(f'This {face_image_path} can be saved to DB for comparison later.')
        output_image_paths[index] = face_image_path
    return output_image_paths

def main(args):
    logger.info(f'Arguments passed are, {args}')
    image = args.get('image')
    output = args.get('output')
    image_output_list = encode_face(image, output)
    for index, image_output in enumerate(image_output_list):
        display_images = []
        main_image = cv2.imread(image_output.get('image_file_path'))
        face_image = cv2.imread(image_output.get('face_file_path'))
        vector_as_image = text_to_image(str(image_output.get('vector')))
        display_images = [main_image,face_image,vector_as_image]
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
        cv2.imshow('Face Encode Canvas', canvas)
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
    exit(main(args))
