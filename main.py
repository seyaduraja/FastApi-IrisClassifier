import uvicorn
from fastapi import FastAPI,Request,Form
from character import character 
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import numpy as np 
import pickle 
import pandas as pd 


# create the app object 
app = FastAPI()
templates = Jinja2Templates(directory = 'javascript learning')
app.mount('/static',StaticFiles(directory = 'static'), name = 'static')
pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.get('/')
def show(request:Request):
    return templates.TemplateResponse('index.html',{'request': request,})


@app.post('/predict')
async def predictspecies(data:character):
    data = data.dict()
    SepalLengthCm = data['SepalLengthCm']
    SepalWidthCm = data['SepalWidthCm']
    PetalLengthCm = data['PetalLengthCm']
    PetalWidthCm = data['PetalWidthCm']
    prediction = classifier.predict([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
    if(prediction[0] == 0):
        prediction = 'Iris-setosa'
    elif(prediction[0] == 1):
        prediction = 'Iris-versicolor'
    else:
        prediction = 'Iris-virginica'
    return{ 
        "prediction": prediction
        }
        
    

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)