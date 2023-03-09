import argparse

import cv2
import dlib
import numpy as np

from image_face_detection import writeOutput
from utils.logger import logger


def face_alignment(image, output, image_upscale=1):
    output_paths = []
    image_to_detect = cv2.imread(image)
    face_detection_classifier = dlib.get_frontal_face_detector()
    all_face_locations = face_detection_classifier(image_to_detect, image_upscale)
    logger.info(f'There are {len(all_face_locations)} faces in image {image}')
    face_landmarks = dlib.full_object_detections()
    face_shape_predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')
    for index, current_face_location in enumerate(all_face_locations):
        # looping through all face detections and append shape predictions
        face_landmarks.append(face_shape_predictor(image_to_detect, current_face_location))
    all_face_chips = dlib.get_face_chips(image_to_detect, face_landmarks)
    for index, current_face_chip in enumerate(all_face_chips):
        if output:
            output_paths.append(writeOutput(output, image, (index + 1), current_face_chip))
    return output_paths


def main(args):
    logger.info(f'Arguments passed are, {args}')
    output_paths = []
    display_images = []
    image = args.get('image')
    output = args.get('output')
    display_images.append(cv2.imread(image))
    output_paths = face_alignment(image, output)
    for output_face in output_paths:
        display_images.append(cv2.imread(output_face.get('face_file_path')))
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
    cv2.imshow('Face Alignment Canvas', canvas)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to directory of images for detecting")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-u", "--image_upscale", type=int, default=1,
                help="Number of times to upscale input image for better output")
ap.add_argument("-o", "--output", required=False,
                help="path to output directory to save output files")

# Main method starts here.
if __name__ == '__main__':
    args = vars(ap.parse_args())
    exit(main(args))
