import configparser
import traceback
from typing import List
from typing import Optional
from pydantic import BaseModel

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from face_recognition_controllers import getFaceVector, matchFaceVectors

# Load environment variables from .env file

configFile = './conf/app.ini'
config = configparser.ConfigParser()
config.read(configFile)

app = FastAPI()


@app.get("/")
def read_root():
    return {"msg": "I am face-recognition rest api"}


@app.get("/api/faceRecognition/vector")
def getFaceVectorController(image_path: str):
    try:
        faceVector = getFaceVector(str.strip(image_path))
    except ValueError as e:
        error = {"status": "failure",
                 "msg": str(e),
                 "data": None
                 }
        return JSONResponse(content=error, status_code=400)
    except Exception as e:
        traceback.print_exc()
        error = {"status": "failure",
                 "msg": "Internal Server error",
                 "data": None
                 }
        return JSONResponse(content=error, status_code=500)
    else:
        return {"status": "success",
                "msg": f"I will return face-vector for input image at {image_path}",
                "data": faceVector.tolist()
                }


class FaceMatch(BaseModel):
    vector1: List[float]
    vector2: List[float]
    threshold: Optional[float]


@app.post("/api/faceRecognition/match")
def matchFaces(body: FaceMatch):
    try:
        vector1 = body.vector1
        vector2 = body.vector2
        threshold = body.threshold
        threshold = threshold or config.getfloat('FACE_RECOGNITION', 'MATCH_THRESHOLD', fallback=0.6)
        match = matchFaceVectors(np.array(vector1), np.array(vector2), threshold=threshold)
        return {"status": "success",
                "msg": f"Input vectors {'matched' if match else 'not matched'} at threshold {threshold}",
                "data": {
                    "match": match,
                    "vector1": vector1,
                    "vector2": vector2
                }
                }
    except Exception as e:
        traceback.print_exc()
        error = {"status": "failure",
                 "msg": "Internal Server error",
                 "data": None
                 }
        return JSONResponse(content=error, status_code=500)


# Start the server
if __name__ == '__main__':
    port = config.getint('UVICORN', 'PORT', fallback='8080')
    workers = config.getint('UVICORN', 'WORKERS', fallback='2')
    log_config = config.get('UVICORN', 'LOG_CONFIG', fallback='')
    uvicorn.run(app="main:app", host="0.0.0.0", port=port, workers=workers, reload=True, log_config=log_config)
