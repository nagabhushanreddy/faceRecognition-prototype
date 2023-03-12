import cv2
import dlib

from utils.logger import logger
from utils.image_utils import *
import face_recognition
import numpy as np

# Scoring function for faces
def score_face(face_location):
    top, right, bottom, left = face_location
    return (right - left) * (bottom - top)


def getSignificantFace(image_path: str, model="hog"):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model=model)
    most_significant_face = max(face_locations, key=score_face)
    top, right, bottom, left = most_significant_face
    face_image = image[top:bottom, left:right]
    return face_image


def getAlignedFace(face_image, model="hog"):
    image_upscale= 1
    cv2_face_image = np.array(face_image, dtype='uint8')
    face_landmarks = dlib.full_object_detections()
    face_shape_predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')
    face_detection_classifier = dlib.get_frontal_face_detector()
    all_face_locations = face_detection_classifier(cv2_face_image, image_upscale)
    for index, current_face_location in enumerate(all_face_locations):
        # looping through all face detections and append shape predictions
        face_landmarks.append(face_shape_predictor(cv2_face_image, current_face_location))
    all_face_chips = dlib.get_face_chips(face_image, face_landmarks)
    alignedFace = face_image
    for index, current_face_chip in enumerate(all_face_chips):
        alignedFace = current_face_chip
    face_image = cv2.cvtColor(alignedFace, cv2.COLOR_BGR2RGB)
    return face_image


def getFaceFeatureVector(face_image, model="hog"):
    face_locations = face_recognition.face_locations(face_image, model=model)
    face_encodings = face_recognition.face_encodings(face_image, face_locations)
    for face_encoding in face_encodings:
        return face_encoding


def getFaceVector(image_path: str, model="hog"):
    try:
        if not is_image(image_path):
            raise ValueError(f"Path {image_path} provided is not a valid image")
        faceImage = getSignificantFace(image_path=image_path, model=model)
        faceImage = getAlignedFace(faceImage, model=model)
        faceVector = getFaceFeatureVector(faceImage, model=model)
    # code that might raise an exception
    except Exception as e:
        logger.error(e)
        raise e
    else:
        return faceVector
    finally:
        pass
    return 0


def get_euclidean_distance(face1_vector, face2_vector):
    # Load the two feature vectors
    feature_vector1 = np.array(face1_vector)
    feature_vector2 = np.array(face2_vector)
    # Calculate the Euclidean distance between the two feature vectors
    euclidean_distance = np.sqrt(np.sum((feature_vector1 - feature_vector2) ** 2))
    return euclidean_distance

def matchFaceVectors(face1_vector, face2_vector, threshold=0.6):
    distances = face_recognition.face_distance([face1_vector], face2_vector)
    # Print the result
    if distances[0] < threshold:
        return True #Matched
    else:
        return False #"Not matched!"
    return 0
