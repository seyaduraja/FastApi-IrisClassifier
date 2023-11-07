from pydantic import BaseModel

class character(BaseModel):
    SepalLengthCm : float 
    SepalWidthCm : float 
    PetalLengthCm : float 
    PetalWidthCm :  float 
    