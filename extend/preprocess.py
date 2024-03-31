from PIL import Image
import numpy

class Prepro():
    def __init__(self,FilePath):
        self.Img = numpy.array(Image.open(FilePath)) 
        self.height = self.Img.shape[0]
        self.width = self.Img.shape[1]
    def Channel_redution(self):
        self.Img = self.Img.sum(axis=2)

    def Convo(self):
        if len(self.Img) == 3:
            Channel_redution(self.Img)
        
        
