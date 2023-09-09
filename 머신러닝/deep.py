from keras.models import Sequential 
from keras.layers import Dense
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