import json
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
img_height = 24
img_width = 24
train_data_dir = 'train'
validation_data_dir = 'val'
nb_train_samples = 18378
nb_validation_samples = 670
batch_size = 128
num_classes = 6

input_shape = (img_height,img_width,3)

tb_callback = TensorBoard()
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size = (img_height, img_width),
                batch_size = batch_size, 
                class_mode = "categorical")
print train_generator.class_indices
val_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


val_generator = val_datagen.flow_from_directory(
                validation_data_dir,
		
                target_size = (img_height, img_width),
                batch_size = batch_size, 
                class_mode = "categorical")

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(128))
#model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax',name='predictions'))

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

json_string = model.to_json()
with open('modelcfg.json','w') as fo:
    fo.write(json_string)

print model.summary()
model.fit_generator(
                train_generator,
                steps_per_epoch =nb_train_samples/batch_size ,
                epochs = 50,
                validation_data = val_generator,
                validation_steps = nb_validation_samples/batch_size,
                max_q_size=16,
                workers=6,
                callbacks=[tb_callback,checkpointer])


model.save('thomas.h5')

