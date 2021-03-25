import tensorflow as tf
from UNet_MSOF_model import Nest_Net2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dataset import dataset_train, dataset_val
input_shape = [256, 256, 6]
model = Nest_Net2(input_shape)
EPOCHS = 10
checkpoint_filepath = './checkpoints'
print(dataset_train.element_spec)
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss')

model.fit(dataset_train.take(1),
    validation_data=dataset_val,
    epochs=EPOCHS,
    callbacks=[model_checkpoint_callback]
)

#model.save('saved_model')