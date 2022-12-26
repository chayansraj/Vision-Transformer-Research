# %%
import pandas as pd 
import numpy as np 
import os
import cv2
import matplotlib.pyplot as plt
import warnings
import scipy

import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')
from tensorflow import keras 
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.inception_v3 import InceptionV3 # InceptionV3


# Set the seed value for experiment reproduci.bility.
seed = 1842
tf.random.set_seed(seed)
np.random.seed(seed)
# Turn off warnings for cleaner looking notebook
warnings.simplefilter('ignore')

# %%
#define image dataset 
#why do we rescale?
image_generator = ImageDataGenerator(rescale=1/255, validation_split=0, shear_range =.25, zoom_range =.2, horizontal_flip = True, rotation_range=20)     


train_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory='/local/data1/chash345/train',
                                                 shuffle=True,
                                                 target_size=(299,299), 
                                                 class_mode='binary')

image_generator = ImageDataGenerator(rescale=1/255) 

validation_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory='/local/data1/chash345/valid',
                                                 shuffle=False,
                                                 target_size=(299,299),
                                                 class_mode='binary')


test_dataset = image_generator.flow_from_directory(
                                                 directory='/local/data1/chash345/test',
                                                 shuffle=False,
                                                 target_size=(299,299), 
                                                 class_mode=None)

# %%
#sns.set_style('white')
generated_image, label = train_dataset.__getitem__(20)
plt.imshow(generated_image[7])

plt.colorbar()
plt.title('Raw femoral fracture X Ray Image')

print(f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height, three single color channel.")
print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")
print(f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}")

# %%
inceptionv3 = InceptionV3(input_shape= (299,299,3), include_top=False, weights= 'imagenet' )
# %%
for layer in inceptionv3.layers:
    layer.trainable = False
#x = Flatten()(model_inception.output)

tf.random.set_seed(150)

model = tf.keras.models.Sequential([
    inceptionv3,
    GlobalAveragePooling2D(),    
    Dense(512,activation="relu"),
    Dropout(0.4),
    Dense(256,activation="relu"),
    Dropout(0.4),
    Dense(128, activation='swish'),  
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(),
              loss="binary_crossentropy",
              metrics=['accuracy'])

# %%
model.summary()

# %%
from sklearn.utils import class_weight
weights = class_weight.compute_class_weight(class_weight= 'balanced', y =train_dataset.classes, classes=np.unique(train_dataset.classes))
dict_weights = {0: weights[0], 1:weights[1]}
dict_weights

# %%
history = model.fit( train_dataset, 
validation_data= validation_dataset, 
epochs= 20, 
verbose=1, 
class_weight=dict_weights
)

# %%
# save the model weights after training
model = model.save('saved_model')

# %%
# Load the saved model anytime for inference
reconstructed_model = keras.models.load_model("saved_model")

# %%
# Predict classes from this reconstructed model
predcited_classes = reconstructed_model.predict_classes(test_dataset)

# %%
# Predict class probabilities from this reconstructed model
predicted_probs = reconstructed_model.predict(test_dataset)

# %%
# %%
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, auc

# %%
fpr, tpr, thresholds = roc_curve(test_dataset.classes, predcited_classes)

# %%
# %%
roc_auc_score(test_dataset.classes, predicted_probs )


# %%
# %%
roc_auc_score(test_dataset.classes, predcited_classes )

# %%
roc_auc = auc(fpr, tpr)

# %%
display = RocCurveDisplay(fpr=fpr,tpr=tpr, roc_auc=roc_auc)
display.plot()
plt.show()

# %%



