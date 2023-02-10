import face_recognition

embedding1 = [0.1, 0.2, 0.3, ..., 0.128]
embedding2 = [0.1, 0.2, 0.3, ..., 0.128]

distance = face_recognition.face_distance([embedding1], embedding2)

print(distance)
