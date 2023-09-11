from keras.models import Sequential 
from keras.layers import Dense, Flatten
from keras.datasets import fashion_mnist 
import numpy as np
import matplotlib.pyplot as plt

##Fashion MNIST 데이터셋 불러오기
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 데이터 시각화
plt.figure() 
plt.imshow(train_images[0]) 
plt.colorbar() 
plt.grid(False)
plt.figure(figsize=(10,10)) 


for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary) 
    plt.xlabel(class_names[train_labels[i]])

# 데이터 스케일링
train_images = train_images / 255.0 
test_images = test_images / 255.0

# 모델 구성
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, batch_size=64, verbose=1)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images) 

print(predictions[0]) 
print(np.argmax(predictions[0]))


# Epoch 1/10
# 938/938 [==============================] - 1s 707us/step - loss: 0.5194 - accuracy: 0.8189
# Epoch 2/10
# 938/938 [==============================] - 1s 685us/step - loss: 0.3864 - accuracy: 0.8626
# Epoch 3/10
# 938/938 [==============================] - 1s 681us/step - loss: 0.3481 - accuracy: 0.8751
# Epoch 4/10
# 938/938 [==============================] - 1s 685us/step - loss: 0.3254 - accuracy: 0.8827
# Epoch 5/10
# 938/938 [==============================] - 1s 680us/step - loss: 0.3063 - accuracy: 0.8885
# Epoch 6/10
# 938/938 [==============================] - 1s 683us/step - loss: 0.2910 - accuracy: 0.8934
# Epoch 7/10
# 938/938 [==============================] - 1s 682us/step - loss: 0.2786 - accuracy: 0.8979
# Epoch 8/10
# 938/938 [==============================] - 1s 679us/step - loss: 0.2664 - accuracy: 0.9012
# Epoch 9/10
# 938/938 [==============================] - 1s 688us/step - loss: 0.2564 - accuracy: 0.9051
# Epoch 10/10
# 938/938 [==============================] - 1s 699us/step - loss: 0.2484 - accuracy: 0.9086
# 313/313 [==============================] - 0s 384us/step - loss: 0.3421 - accuracy: 0.8780
# Test accuracy: 0.878000020980835
# 313/313 [==============================] - 0s 325us/step
# [3.1202458e-07 2.4531482e-07 5.2561379e-08 3.0678333e-07 1.3223049e-08
#  2.3275001e-03 1.2553801e-06 3.9960749e-02 2.3636862e-06 9.5770717e-01]
# 9