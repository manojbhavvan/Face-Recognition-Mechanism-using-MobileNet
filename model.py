import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import datetime
import time

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sample_image = image.load_img("C:/Users/JASWANTH/OneDrive/Desktop/FR Dataset/train/jennie/jennie1.jpg")
plt.imshow(sample_image)

batch_size = 10
height = 224
width = 224
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/JASWANTH/OneDrive/Desktop/FR Dataset/train/',
    validation_split=0.2,
    subset = 'training',
    seed=123,
    image_size=(height,width),
    batch_size=batch_size
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/JASWANTH/OneDrive/Desktop/FR Dataset/validation/',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(height, width),
    batch_size=batch_size)

classes = np.array(train_dataset.class_names)
print(classes)

norm_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_data = train_dataset.map(lambda x, y: (norm_layer(x), y))
vali_data = validation_dataset.map(lambda x,y: (norm_layer(x),y))
autotune = tf.data.AUTOTUNE
train_data = train_data.cache().prefetch(buffer_size=autotune)
vali_data = vali_data.cache().prefetch(buffer_size=autotune)

feature_extractor_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
feature_layer = hub.KerasLayer(
    feature_extractor_model,
    input_shape=(224,224,3),
    trainable=False)
num_classes = len(classes)
model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(num_classes)])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

log_dir = 'logs/fit/'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1)

history = model.fit(train_data,validation_data=vali_data,epochs=10,callbacks=tensorboard_callback)

from keras.preprocessing.image import load_img,img_to_array
image = load_img('C:/Users/Admin/Desktop/FR Dataset/test/test2.jpg', target_size = (height,width))
image = img_to_array(image)/255.0
image = np.expand_dims(image , axis=0)
images = np.vstack([image])

predicted_batch = model.predict(images)
predicted_id = tf.math.argmax(predicted_batch, axis=-1)
predicted_label = classes[predicted_id]
print(predicted_label)
#Predicting the image trainned for a single class need to improve it for multiple class
#Delete the comments after changing the part


sample = cv2.imread("C:/Users/JASWANTH/OneDrive/Desktop/FR Dataset/test/d03.jpg")
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
boxes = classifier.detectMultiScale(sample, 1.3, 3)


for box in boxes:
    x, y, width, height = box
    x2, y2 = x + width, y + height
    cv2.rectangle(sample, (x, y), (x2, y2), (0, 0, 255), 1)

cv2.imshow('Face Recognition', sample)
cv2.waitKey(0)

import xlwt
from xlwt import Workbook
import numpy as np

name_list=np.array(['jennie','jisoo', 'lisa', 'rose'])



wb=Workbook()
style = xlwt.easyxf('font: bold 1')
sheet = wb.add_sheet('Day 1')


j=1
sheet.write(0,0,'Sl.No',style)
sheet.write(0,1,'Members',style)
sheet.write(0,2,'P/A',style)
for i in range(len(classes)):
    sheet.write(j,0,i) #sno
    j+=1
i=0
while(i<4):
    if(classes[i]==name_list[i]):
        sheet.write(i+1,1,classes[i])
        sheet.write(i+1,2,'P')
        i+=1
    else:
        sheet.write(i+1,1,classes[i])
        sheet.write(i+1,2,'A')
        i+=1

wb.save('Attendance_Sheet.xls')
