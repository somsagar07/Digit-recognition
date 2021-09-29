import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
'''
mnist = tf.keras.datasets.mnist
(train1, train2), (test1 , test2) = mnist.load_data()

train1 = tf.keras.utils.normalize(train1, axis=1)
test1 = tf.keras.utils.normalize(test1, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train1, train2, epochs=10)

model.save('hdm')
'''
model = tf.keras.models.load_model('hdm')

while os.path.isfile(f"number.png"):
    img=cv2.imread(f"number.png")[:,:,0]
    img=np.invert(np.array([img]))
    predict=model.predict(img)
    print(f"digit is {np.argmax(predict)}")
    break
