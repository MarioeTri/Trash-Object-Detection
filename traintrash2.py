import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

dataset_dir = 'dataset-resized'

datagen = ImageDataGenerator(
    rescale=1./255,  
    validation_split=0.2,
    rotation_range=20,  
    width_shift_range=0.2, 
    height_shift_range=0.2,
    shear_range=0.2,  
    zoom_range=0.2,  
    horizontal_flip=True,  
    fill_mode='nearest'  
)

train_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training', 
)

val_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation', 
)

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  

model = models.Sequential([
    base_model,  
    layers.GlobalAveragePooling2D(),  
    layers.BatchNormalization(),  
    layers.Dense(128, activation='relu'),  
    layers.Dense(train_data.num_classes, activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

def scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        lr = lr * 0.5  
    return lr

lr_scheduler = LearningRateScheduler(scheduler)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,  
    callbacks=[early_stopping, lr_scheduler]  
)

model.save('model3.h5')
