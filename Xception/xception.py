# %%
import numpy as np
import os
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.activations import relu, softmax, sigmoid, swish
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input, Xception
from tensorflow.keras.preprocessing import image

# %%
datagenerator_train = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    rotation_range=30,
    vertical_flip=False,
    #brightness_range=[0.90,1.25],
    fill_mode='nearest'
)

datagenerator = ImageDataGenerator(preprocessing_function=preprocess_input)

# %%
# load and iterate training dataset
train_data = datagenerator_train.flow_from_directory('/local/data1/chash345/train', 
    class_mode='binary',
    target_size=(299, 299), 
    batch_size=32, 
    shuffle=False)

# load and iterate validation dataset
val_data = datagenerator.flow_from_directory('/local/data1/chash345/valid', 
    class_mode='binary',
    target_size=(299, 299),
    batch_size=32, 
    shuffle=False)

# load and iterate test dataset
test_data = datagenerator.flow_from_directory('/local/data1/chash345/test', 
    class_mode='binary',
    target_size=(299, 299),
    batch_size=32, 
    shuffle=False)

# %%
#sns.set_style('white')
generated_image, label = train_data.__getitem__(20)
plt.imshow(generated_image[7])

plt.colorbar()
plt.title('Raw Chest X Ray Image')

print(f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height, one single color channel.")
print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")
print(f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}")

# %%
generated_image.shape

# %%
pre_trained_model_xception = Xception(
    input_shape=(299,299,3),
    include_top=False,
    weights="imagenet")

# %%
# Some weights in later layers are unfreezed
for layer in pre_trained_model_xception.layers[:-5]:
    layer.trainable=False

tf.random.set_seed(10)

inputs = keras.Input(shape=(299,299,3))
norm_layer = keras.layers.experimental.preprocessing.Normalization()
mean = np.array([127.5]*3)
var = mean ** 2
x = norm_layer(inputs)
norm_layer.set_weights([mean , var])

x = pre_trained_model_xception(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128,activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)


# model = tf.keras.models.Sequential([
#     pre_trained_model_xception,
#     Flatten(),    
#     Dense(256,activation="swish"),
#     Dropout(0.4),
#     Dense(256,activation="swish"),
#     Dropout(0.4),
#     Dense(128, activation='swish'),  
#     Dense(1, activation='sigmoid')
# ])

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

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


