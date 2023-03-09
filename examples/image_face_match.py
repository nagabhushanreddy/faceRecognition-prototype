import argparse
import cv2
import face_recognition
import numpy as np

from image_face_encoding import encode_face
from utils.logger import logger
from utils.text_to_image import text_to_image

def get_euclidean_distance(face1_vector, face2_vector):
    # Load the two feature vectors
    feature_vector1 = np.array(face1_vector)
    feature_vector2 = np.array(face2_vector)
    # Calculate the Euclidean distance between the two feature vectors
    euclidean_distance = np.sqrt(np.sum((feature_vector1 - feature_vector2) ** 2))
    return euclidean_distance


def is_match(face1_vector, face2_vector, match_threshold=0.8):
    euclidean_distance = get_euclidean_distance(face1_vector, face2_vector)
    logger.info(f'Euclidean distance is {euclidean_distance}')
    if euclidean_distance < (1 - match_threshold):
        return True
    else:
        return False


def main(args):
    logger.info(f'Arguments passed are, {args}')
    image1 = args.get('image1')
    image2 = args.get('image2')
    output = args.get('output')
    threshold = args.get('threshold')
    face1_object = encode_face(image1, output)
    face2_object = encode_face(image2, output)

    face_distance = face_recognition.face_distance(np.array(face1_object[0].get('vector')),
                                                   np.array(face2_object[0].get('vector'))
                                                   )
    logger.info(f'Face distance {face_distance}')

    if is_match(face1_object[0].get('vector'), face2_object[0].get('vector'), threshold):
        logger.info(f'{image1} and {image2} are a match!')
    else:
        logger.info(f'Oops! {image1} and {image2} are not same!')
    return 0


from face_recognition import api
from face_recognition import face_recognition_cli
from face_recognition import face_detection_cli
import os


def main2(args):
    image1 = args.get('image1')
    image2 = args.get('image2')
    threshold = args.get('threshold')
    img_a1 = api.load_image_file(image1)
    img_a2 = api.load_image_file(image2)
    face_encoding_a1 = api.face_encodings(img_a1)[0]
    face_encoding_a2 = api.face_encodings(img_a2)[0]
    faces_to_compare = [
        face_encoding_a1]
    match_results = api.compare_faces(faces_to_compare, face_encoding_a2, tolerance=(1 - threshold))
    distance_results = api.face_distance(faces_to_compare, face_encoding_a2)
    logger.info(f"Matched: {match_results[0]}, Distance: {distance_results[0]}")

    match_distance_image = text_to_image(f'\n' +
                                         f'Distance = {distance_results[0]}'
                                         f'\n' +
                                         f'Matched = {match_results[0]}'
                                         f'\n')

    image1_display = cv2.imread(image1)
    image2_display = cv2.imread(image2)
    display_images = [image1_display, match_distance_image, image2_display,]
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
    cv2.imshow('Face Match Canvas', canvas)
    cv2.waitKey(20000)
    cv2.destroyAllWindows()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--image1", required=True,
                help="path to directory of image1 for comparing")
ap.add_argument("-i2", "--image2", required=False,
                help="path to directory of image2 for comparing")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
ap.add_argument(
    "-t",
    "--threshold",
    required=True,
    type=float,
    choices=[
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9],
    help="Threshold to reject images with almost same size faces")
ap.add_argument("-o", "--output", required=False,
                help="path to output directory to save output files")

# Main method starts here.
if __name__ == '__main__':
    args = vars(ap.parse_args())
    exit(main2(args))
