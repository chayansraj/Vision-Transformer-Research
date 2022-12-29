# %%
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.activations import relu, softmax, sigmoid, swish
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50  import preprocess_input

# %%
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, vertical_flip=True )
validation_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

# %%
# load and iterate training dataset
train_data = train_datagen.flow_from_directory('/local/data1/chash345/train', 
    class_mode='binary',
    target_size=(224, 224), 
    batch_size=32, 
    shuffle=False,
    color_mode='rgb'
)

# load and iterate validation dataset
val_data = validation_datagen.flow_from_directory('/local/data1/chash345/valid', 
    class_mode='binary',
    target_size=(224, 224),
    batch_size=32, 
    shuffle=False,
    color_mode='rgb'
)

# load and iterate test dataset
test_data = test_datagen.flow_from_directory('/local/data1/chash345/test', 
    class_mode='binary',
    target_size=(224, 224),
    batch_size=8, 
    shuffle=False,
    color_mode='rgb'
)

# %%
#sns.set_style('white')
generated_image, label = train_data.__getitem__(24)
plt.imshow(generated_image[7])

plt.colorbar()
plt.title('Raw Chest X Ray Image')

print(f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height, one single color channel.")
print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")
print(f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}")

# %%
generated_image.shape

# %%
pre_trained_model = VGG19(input_shape=(224,224,3),
                                include_top=False,
                                weights="imagenet")

# Some weights in later layers are unfreezed
for layer in pre_trained_model.layers[:-5]:
    layer.trainable=False

tf.random.set_seed(100)

model = tf.keras.models.Sequential([
    pre_trained_model,
    Flatten(),    
    Dense(512,activation="swish"),
    Dropout(0.5),
    # Dense(256,activation="relu"),
    # Dropout(0.5),
    Dense(128, activation='relu'),  
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(learning_rate=1e-4),
              loss="binary_crossentropy",
              metrics=['accuracy'])

# %%
model.summary()

# %%
from sklearn.utils import class_weight
weights = class_weight.compute_class_weight(class_weight= 'balanced', y =train_data.classes, classes=np.unique(train_data.classes))
dict_weights = {0: weights[0], 1:weights[1]}
dict_weights

# %%
history = model.fit(
    train_data,
    epochs=20,
    validation_data=val_data, 
    class_weight=dict_weights 
)

# %%
model = model.save('saved_model')

# %%
reconstructed_model = keras.models.load_model("vgg16")

# %%
predcited_classes = reconstructed_model.predict_classes(test_data)

# %%
predicted_probs = reconstructed_model.predict(test_data)

# %%
from sklearn.metrics import roc_auc_score, roc_curve

# %%
fpr, tpr, thresholds = roc_curve(test_data.classes, predcited_classes)

# %%
roc_auc_score(test_data.classes, predicted_probs )

# %%
roc_auc_score(test_data.classes, predcited_classes )

# %%
plt.plot(fpr,tpr)

# %%



