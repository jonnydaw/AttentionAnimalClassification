import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential, Model
import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, MultiHeadAttention, Flatten, Rescaling, Dropout, Concatenate, Add,Flatten
from keras import regularizers
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score

class MyThresholdCallback(keras.callbacks.Callback):
    #https://stackoverflow.com/questions/59563085/how-to-stop-training-when-it-hits-a-specific-validation-accuracy
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold
        self.accs = []

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs["val_accuracy"]
        self.accs.append(val_acc)
        if val_acc >= self.threshold or (len(set(self.accs[-6:])) == 1 and len(self.accs) > 12): 
            self.model.stop_training = True      

# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory

data_dir = "./predict_all_images"
training_data = keras.utils.image_dataset_from_directory(
    data_dir,
    shuffle=False
)

global_img_data_X = []
global_img_data_y = []
first_partial_img_data_X = []
first_partial_img_data_y = []
second_partial_img_data_X = []
second_partial_img_data_y = []
third_partial_img_data_X = []
third_partial_img_data_y = []

# https://stackoverflow.com/questions/56226621/how-to-extract-data-labels-back-from-tensorflow-dataset
img_numps = np.concatenate([x for x, y in training_data], axis=0)

labels = np.concatenate([y for x, y in training_data], axis=0)
print(labels) 

for i in range(len(img_numps)):
    current_img_nump = img_numps[i]
    global_img_data_X.append(current_img_nump)
    global_img_data_y.append(labels[i])

   # https://www.geeksforgeeks.org/splitting-arrays-in-numpy/
    splits = np.array_split(current_img_nump, 3)
    first_partial_img_data_X.append(splits[0])
    first_partial_img_data_y.append(labels[i])
    second_partial_img_data_X.append(splits[1])
    second_partial_img_data_y.append(labels[i])
    third_partial_img_data_X.append(splits[2])
    third_partial_img_data_y.append(labels[i])


global_img_data_X = np.array(global_img_data_X)
global_img_data_y = np.array(global_img_data_y)
first_partial_img_data_X = np.array(first_partial_img_data_X)
first_partial_img_data_y = np.array(first_partial_img_data_y)
second_partial_img_data_X = np.array(second_partial_img_data_X)
second_partial_img_data_y = np.array(second_partial_img_data_y)
third_partial_img_data_X = np.array(third_partial_img_data_X)
third_partial_img_data_y = np.array(third_partial_img_data_y)

#https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
img_arr = global_img_data_X[31]
img_arr = img_arr.astype(np.uint8)

# done just to verify
image = Image.fromarray(img_arr)
image.show()
print('global label',global_img_data_y[31])

img_arr = first_partial_img_data_X[31]
img_arr = img_arr.astype(np.uint8)
image = Image.fromarray(img_arr)
image.show()
print('first label',first_partial_img_data_y[31])

img_arr = second_partial_img_data_X[31]
img_arr = img_arr.astype(np.uint8)
image = Image.fromarray(img_arr)
image.show()
print('second label',second_partial_img_data_y[31])

img_arr = third_partial_img_data_X[31]
img_arr = img_arr.astype(np.uint8)
image = Image.fromarray(img_arr)
image.show()
print('third label',third_partial_img_data_y[31])
class_names = training_data.class_names

num_classes = len(class_names)

# https://www.tensorflow.org/tutorials/images/classification

def simple_model_builder():
    model = Sequential()
    model.add(Rescaling(1./255))
    model.add(Conv2D(16, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dense(num_classes))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
    
# https://nrl.northumbria.ac.uk/id/eprint/39658/1/Wu%20et%20al%20-%20Audio%20classification%20using%20attention-augmented%20convolutional%20neural%20network%20AAM.pdf    
def model_builder():
    top_in = keras.Input(shape=(86, 256, 3))
    middle_in = keras.Input(shape=(85, 256, 3))
    bottom_in = keras.Input(shape=(85, 256, 3))
    glob_in = keras.Input(shape=(256, 256, 3))

    top_branch = Sequential()
    top_branch.add(Rescaling(1./255))
    top_branch.add(Conv2D(16, 3, padding='same', activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)))
    top_branch.add(MaxPooling2D())
    top_branch.add(Conv2D(32, 3, padding='same', activation='relu'))
    top_branch.add(MaxPooling2D())
    top_branch.add(Conv2D(64, 3, padding='same', activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)))
    top_branch.add(MaxPooling2D())
    # dropout shouldn't be too high for cnn
    top_branch.add(Dropout(0.1))

    middle_branch = Sequential()
    middle_branch.add(Rescaling(1./255))
    middle_branch.add(Conv2D(16, 3, padding='same', activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)))
    middle_branch.add(MaxPooling2D())
    middle_branch.add(Conv2D(32, 3, padding='same', activation='relu'))
    middle_branch.add(MaxPooling2D())
    middle_branch.add(Conv2D(64, 3, padding='same', activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)))
    middle_branch.add(MaxPooling2D())
    middle_branch.add(Dropout(0.1))

    bottom_branch = Sequential()
    bottom_branch.add(Rescaling(1./255))
    bottom_branch.add(Conv2D(16, 3, padding='same', activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)))
    bottom_branch.add(MaxPooling2D())
    bottom_branch.add(Conv2D(32, 3, padding='same', activation='relu'))
    bottom_branch.add(MaxPooling2D())
    bottom_branch.add(Conv2D(64, 3, padding='same', activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)))
    bottom_branch.add(MaxPooling2D())
    bottom_branch.add(Dropout(0.1))

    glob = Sequential()
    glob.add(Rescaling(1./255))
    glob.add(Conv2D(16, 3, padding='same', activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)))
    glob.add(MaxPooling2D())
    glob.add(Conv2D(32, 3, padding='same', activation='relu'))
    glob.add(MaxPooling2D())
    # has to be 60 otherwise shapes are mismatched
    glob.add(Conv2D(60, 3, padding='same', activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)))
    glob.add(MaxPooling2D())
    glob.add(Dropout(0.1))


    print("*******************************")
    # https://stackoverflow.com/questions/47678108/keras-use-one-model-output-as-another-model-input
    top_output = top_branch(top_in)
    print(top_output.shape)

    middle_output = middle_branch(middle_in)
    print(middle_output.shape)

    bottom_output = bottom_branch(bottom_in)
    print(bottom_output.shape)

    glob_output = glob(glob_in)
    print(glob_output.shape)

    print("*******************************")

    # verify shapes
    print(top_output.shape)
    print(middle_output.shape)
    print(bottom_output.shape)
    print(glob_output.shape)

    # higher num_heads would be better, not possible on my machine due to mem. constraints
    attention_top = MultiHeadAttention(num_heads=3, key_dim=64)(top_output, top_output)
    attention_mid = MultiHeadAttention(num_heads=3, key_dim=64)(middle_output, middle_output)
    attention_bot = MultiHeadAttention(num_heads=3, key_dim=64)(bottom_output, bottom_output)
    attention_glob = MultiHeadAttention(num_heads=3, key_dim=64)(glob_output,glob_output)

    attention_top = Flatten()(attention_top)
    attention_mid = Flatten()(attention_mid)    
    attention_bot = Flatten()(attention_bot)
    attention_glob = Flatten()(attention_glob)

    concatenated = Concatenate()([attention_top, attention_mid, attention_bot])
    addition = Add()([attention_glob.output, concatenated])
    x = Dense(num_classes, activation='softmax')(addition)


    model = Model(inputs=[top_in, middle_in, bottom_in, glob_in], outputs=x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

X_train, X_test, y_train, y_test = train_test_split(global_img_data_X, global_img_data_y, test_size=0.4)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)
print(X_test.shape, X_val.shape, y_test.shape, y_val.shape)

X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(first_partial_img_data_X, first_partial_img_data_y, test_size=0.4)
print(X_train_top.shape, X_test_top.shape, y_train_top.shape, y_test_top.shape)
X_test_top, X_val_top, y_test_top, y_val_top = train_test_split(X_test_top, y_test_top, test_size=0.5)
print(X_train_top.shape, X_val_top.shape, y_test_top.shape, y_val_top.shape)

X_train_mid, X_test_mid, y_train_mid, y_test_mid = train_test_split(second_partial_img_data_X, second_partial_img_data_y, test_size=0.4)
print(X_train_mid.shape, X_test_mid.shape, y_train_mid.shape, y_test_mid.shape)
X_test_mid, X_val_mid, y_test_mid, y_val_mid = train_test_split(X_test_mid, y_test_mid, test_size=0.5)
print(X_test_mid.shape, X_val_mid.shape, y_test_mid.shape, y_val_mid.shape)

X_train_bot, X_test_bot, y_train_bot, y_test_bot = train_test_split(third_partial_img_data_X, third_partial_img_data_y, test_size=0.4)
print(X_train_bot.shape, X_test_bot.shape, y_train_bot.shape, y_test_bot.shape)
X_test_bot, X_val_bot, y_test_bot, y_val_bot = train_test_split(X_test_bot, y_test_bot, test_size=0.5)
print(X_test_bot.shape, X_val_bot.shape, y_test_bot.shape, y_val_bot.shape)

callback = MyThresholdCallback(threshold=0.95)

simple_model = simple_model_builder()
history_simple = simple_model.fit(X_train,y_train,epochs=45, validation_data = (X_val,y_val), callbacks = [callback])
history_dict_simple = history_simple.history
acc_simple = history_dict_simple['accuracy']
val_acc_simple = history_dict_simple['val_accuracy']
loss_simple = history_dict_simple['loss']
val_loss_simple = history_dict_simple['val_loss']
epochs_simple = range(1, len(acc_simple) + 1)

y_pred_simple = simple_model.predict(X_val)
y_pred_classes_simple = np.argmax(y_pred_simple, axis=1)


plt.plot(epochs_simple, val_acc_simple, label='Validation accuracy', color='blue')
plt.plot(epochs_simple, acc_simple, label='Training accuracy',color='orange')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(epochs_simple, loss_simple, label='Loss', color = 'red')
plt.plot(epochs_simple, val_loss_simple, label='Validation Loss', color = 'purple')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



print("shapesshapesshapesshapesshapesshapesshapesshapes")
print(X_train_top.shape, X_train_mid.shape, X_train_bot.shape, X_train.shape)
model = model_builder()
history = model.fit([X_train_top, X_train_mid, X_train_bot, X_train], 
                    y_train, 
                    epochs=45,
                    validation_data=([X_val_top, X_val_mid, X_val_bot, X_val], y_val),
                    callbacks=[callback])

model.save("cnn_att_20_05_24_save.keras")
history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)


print("shapesshapesshapesshapesshapesshapesshapes")
print(X_val_top, X_val_mid, X_val_bot, X_val)
y_pred = model.predict([X_val_top, X_val_mid, X_val_bot, X_val])
y_pred_classes = np.argmax(y_pred, axis=1)



plt.plot(epochs, val_acc, label='Validation accuracy', color='blue')
plt.plot(epochs, acc, label='Training accuracy',color='orange')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(epochs, loss, label='Loss', color = 'red')
plt.plot(epochs, val_loss, label='Validation Loss', color = 'purple')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()