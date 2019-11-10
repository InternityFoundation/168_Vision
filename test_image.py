import cv2
import numpy as np
from PIL import Image
from keras import models
import serial 

#Load the saved model
model = models.load_model('model1.h5')
video = cv2.VideoCapture(0)
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=0.5)
counter = 0
preds = []

#while(counter < 5):
counter = counter + 1
_, frame = video.read()

        #Convert the captured frame into RGB
im = Image.fromarray(frame, 'RGB')

        #Resizing into 128x128 because we trained the model with this image size.
im = im.resize((128,128))
img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3 
img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict method on model to predict 'me' on the image
prediction = int(model.predict(img_array)[0][0])
if(prediction == 1):
    preds.append("Non-Biodegradable")
if(prediction == 0):
    preds.append("Biodegradable")
        #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
#        if prediction == 0:
#            ser.write(b'1')
#            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#            print("Non-Biodegradable")
#        else:
#            ser.write(b'2')
#            print("Biodegradable")
cv2.imshow("Capturing", frame)
key=cv2.waitKey(1)
if key == ord('q'):
    ser.close()
 

count1 = 0
count2 = 0
for x in preds:
    if(x == "Biodegradable"):
        count1 = count1 + 1
    if(x == "Non-Biodegradable"):
        count2 = count2 + 1
    
if(count1 > count2):
    ser.write(b'1')
    print("Biodegradable")
if(count1 < count2):
    ser.write(b'2')
    print("Non-Biodegradable")
    

    
ser.close()    
video.release()
cv2.destroyAllWindows()