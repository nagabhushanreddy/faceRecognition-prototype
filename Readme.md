# Environment

- Anaconda 
- Python 
- FAST API (rest api)
- Face recognition libraries (important below, all libraries in usage for microservice application can be found in environment.yml)
  - OpenCV
  - DLib
  - numpy
  - sckit learn 

# Steps to run microservice

Below steps to be run for bringing-up the microservice.  

## Install Anaconda

[Download-Link](https://www.anaconda.com/products/distribution) 

## Check if installation is successful 

### From Terminal run below command 

```commandline
(base) naga@MAC011 ~ % conda -V
conda 22.11.1
(base) naga@MAC011 ~ %
```

## Install required libraries

Below also installs required python version and creates a python environment named `faceRecognition` 

```commandline
conda env create -f environment.yml
```

Switch to python environment `faceRecognition`  

```commandline
conda activate faceRecognition
```

## Rest-api Microservices 

<table>
    <tr>
        <td>name</td>
        <td>url</td>
        <td>method</td>
        <td>docs</td>
        <td>body</td>
    </tr>
    <tr>
        <td>Get Face Vector (Face Signature)</td>
        <td>/api/faceRecognition/vector?image_path=/Users/naga/workspace/faceRecognition-prototype/images/elon.jpg</td>
        <td>GET</td>
        <td>This api will generate face vector for input image path and returns face-vector as response.</td>
        <td></td>
    </tr>
    <tr>
        <td>Compare Faces for similarity or equality </td>
        <td>/api/faceRecognition/match</td>
        <td>POST</td>
        <td>This api will  compare two vectors at input threshold level and provides match result with in threshold. </td>
        <td>[object Object]</td>
    </tr>
</table>

Note: it is recommended to install [thunder-client](https://www.thunderclient.com/) and go through the [documentation](./test/thunder-collection_Face-Recognition.json) importing into thunder client for through understanding. 

## Start Face-recognition Micro-service 

```commandline
faceRecognition) naga@MAC011 ~ % python ./main.py
```
## Configuration 

- All configuration needed for micro-service application are placed in `./conf` folder
- `app.ini` has application related configuration, a default face distance threshold is maintained here when threshold is not passed as input for rest-api. Another important attribute is port number on which http runs is also placed here and can be changed accordingly.
- Logging configuration is mainted in `logging.ini`
- When configuration is changed, the application should be restarted for changes to be effective. 

## Understand Fundamentals

### Example Scripts

#### Detect Face 
```commandline
python ./examples/image_face_detection.py -i images/DSCF1934.JPG -d hog -o ./output
```

#### Align Face 
```commandline
faceRecognition) naga@MAC011 ~ % python ./examples/image_face_alignment.py -i images/elon.JPG -d hog -o ./output
```

#### Encode Face 
```commandline
(faceRecognition) naga@MAC011 ~ % python ./examples/image_face_encoding.py -i images/elon.jpg -d hog -o ./output
```

#### Match Face / Comparison

```commandline
(faceRecognition) naga@MAC011 ~ % python ./examples/image_face_match.py -i1 images/obama2.jpg -i2 images/obama.jpg -t 0.7

(faceRecognition) naga@MAC011 ~ % python ./examples/image_face_match.py -i1 images/elon.png -i2 images/obama.jpg -t 0.6
```
