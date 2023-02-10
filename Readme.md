# Environment

## Install Anaconda

## Check if installation is successful 

### From Terminal run below command 

```commandline
(base) naga@MAC011 ~ % conda -V
conda 22.11.1
(base) naga@MAC011 ~ %
```

## Install required libraries
```commandline
conda env create -f environment.yml
```

## Scripts 

### Detect Face 
```commandline
python ./image_face_detection.py -i images/DSCF1934.JPG -d hog -o ./output
```

### Align Face 
```commandline
faceRecognition) naga@MAC011 ~ % python ./image_face_alignment.py -i images/elon.JPG -d hog -o ./output
```

### Encode Face 
```commandline
(faceRecognition) naga@MAC011 ~ % python ./image_face_encoding.py -i images/elon.jpg -d hog -o ./output
```

### Match Face / Comparison

```commandline
(faceRecognition) naga@MAC011 ~ % python ./image_face_match.py -i1 images/obama2.jpg -i2 images/obama.jpg -t 0.7

(faceRecognition) naga@MAC011 ~ % python ./image_face_match.py -i1 images/elon.png -i2 images/obama.jpg -t 0.6

```