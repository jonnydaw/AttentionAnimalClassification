from keras import models
from sklearn.model_selection import train_test_split
from  feature_extraction import FeatureExtraction
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
# https://www.geeksforgeeks.org/create-a-voice-recorder-using-python/
import sounddevice as sd
from scipy.io.wavfile import write
from my_utils import MyUtils
from PIL import Image
import numpy as np
import grad_cam as gc
from analysis import eval
import os 
import shutil

def load_and_copy():
    '''
        Loading and copying audio and saving it as a .wav file
        Args:
            None
        Returns:
            None
        Raises:
            None
    '''
    location = input("Enter full audio file location: ")
    location = location.strip()
    if(location.split(".")[1] != "wav"):
        print("Audio has to be .wav")
    else:
        predict_location = "./predict/predict.wav"

        shutil.copy(location, predict_location)
translate = {"Aslan" : "Lion", "Esek" : "Donkey", "Inek": "Cow", "Kedi-Part1" : "Cat", "Kopek-Part1" : "Dog", "Koyun": "Sheep", "Kurbaga": "Frog", "Kus-Part1": "Bird", "Maymun": "Monkey", "Tavuk" :"Chicken"}

class Predictions(FeatureExtraction):
    def __init__(this, sounds_location):
        super().__init__(sounds_location)
        this.X_test = None
        this.y_test = None

    def record_and_save_noise(this):
        '''
            Recording audio and saving it as a .wav file
            Args:
                None
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        sr = 44100
        duration = 8
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=2)
        sd.wait()
        write("./predict/predict.wav", sr, recording)
        

    def scale(this,template):
        '''
            Scaling the predicted values in line with the original
            scaling based on whether the prediction is based on
            the fully balanced dataset or the trimmed one
            Args:
                template (string) : location of the the csv to scale against
            Returns:
                scaled_vals (numpy array???) : scaled values
            Raises:
                Not yet implemented
        '''
        df_prediction = pd.read_csv("po.csv")
        print()
        scale_template = pd.read_csv(template)
        scale_template = scale_template.drop("species",axis=1)      
        df_prediction = df_prediction.drop("species",axis=1)   
        max_vals = scale_template.max(axis=0).values
        min_vals = scale_template.min(axis=0).values
        # print("***************************")
        print(df_prediction.values)
        scaled_vals = (df_prediction.values - min_vals) / (max_vals - min_vals)
        return scaled_vals
    
    def get_test_data_nns(this,csv):
        df = pd.read_csv(csv)
        y = df.iloc[:, 0] 
        X = df[[x for x in df.columns[1:]]]
        _, this.X_test,_,this.y_test = train_test_split(X,y, test_size = 0.3, random_state = 14)

    def predictions(this,*model_options):  
        '''
            Making the prediction/s for the audio file.
            Args:
                *model_options (str) : Variable length argument list made up of model names.
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        valid_options = ['balanced_concat.keras','balanced_single.keras','trimmed_concat.keras','trimmed_single.keras']
        for model_option in model_options:
            print("***********************************************************")
            print(model_option)
            # determine which type of model
            template_type = model_option.split("_")[0]
            if(template_type == 'balanced'):
                this.get_test_data_nns('processed_sound_stats_balanced.csv')
            else:
                this.get_test_data_nns('processed_sound_stats_trimmed.csv')
            if(model_option in valid_options):    
                try:
                    if(model_option == "balanced_concat.keras" or model_option == "trimmed_concat.keras"):
                        model = models.load_model(model_option)
                        
                        x = this.scale(template_type + "_template.csv")
                        # concat has to accept in below format
                        x = [x,x]
                        probs = model.predict(x)
                        print(probs)
                        predicted_class = np.argmax(probs)
                        print("Predicted class index:", predicted_class)
                        factorised_to_original = MyUtils.load_dict(template_type + "_transformation")
                        print("Predicted class: ", translate[factorised_to_original[predicted_class]])
                        eval([this.X_test,this.X_test],this.y_test,model,factorised_to_original)
                        print("****************************************************************")
                    else:
                        model = models.load_model(model_option)
                        print('templatetemplate')
                        print(template_type)
                        x = this.scale(template_type + "_template.csv")
                        probs = model.predict(x)
                        print(probs)
                        predicted_class = np.argmax(probs)
                        print("Predicted class index:", predicted_class)
                        factorised_to_original = MyUtils.load_dict(template_type + "_transformation")
                        print("Predicted class: ", translate[factorised_to_original[predicted_class]])
                        eval(this.X_test,this.y_test,model,factorised_to_original)
                        print("****************************************************************")

                except ValueError:
                    raise("Not a valid model")

    def predict_img(this):
        int_class_to_animal = {0:"Aslan",1:"Esek",2:"Inek",3:"Kedi",4:"Kopek",5:"koyun",6:"kurbaga",7:"kus",8:"maymun",9:"tavuk"}
        global_img_data_X = []
        global_img_data_y = []
        first_partial_img_data_X = []
        first_partial_img_data_y = []
        second_partial_img_data_X = []
        second_partial_img_data_y = []
        third_partial_img_data_X = []
        third_partial_img_data_y = []
        training_data = keras.utils.image_dataset_from_directory(
        "./predict_image",
        shuffle=False
            )

        img_numps = np.concatenate([x for x, y in training_data], axis=0)
        img_nump = img_numps[0]
        print(img_nump.shape)

        splits = np.array_split(img_nump, 3)
        first_partial_img_data_X.append(splits[0])
        second_partial_img_data_X.append(splits[1])
        third_partial_img_data_X.append(splits[2])
        global_img_data_X.append(img_nump)
        global_img_data_X = np.array(global_img_data_X)
        global_img_data_y = np.array(global_img_data_y)
        first_partial_img_data_X = np.array(first_partial_img_data_X)
        first_partial_img_data_y = np.array(first_partial_img_data_y)
        second_partial_img_data_X = np.array(second_partial_img_data_X)
        second_partial_img_data_y = np.array(second_partial_img_data_y)
        third_partial_img_data_X = np.array(third_partial_img_data_X)
        third_partial_img_data_y = np.array(third_partial_img_data_y)
        img_arr = global_img_data_X[0]
        img_arr = img_arr.astype(np.uint8)
        image = Image.fromarray(img_arr)
        # image.show()
        

        img_arr1 = first_partial_img_data_X[0]
        img_arr1 = img_arr1.astype(np.uint8)
        image = Image.fromarray(img_arr1)
        # image.show()
        

        img_arr2 = second_partial_img_data_X[0]
        img_arr2 = img_arr2.astype(np.uint8)
        image = Image.fromarray(img_arr2)
        # image.show()
        

        img_arr3 = third_partial_img_data_X[0]
        print("3 shape", img_arr3.shape)
        img_arr3 = img_arr3.astype(np.uint8)
        image = Image.fromarray(img_arr3)
        # image.show()
        
        model = models.load_model('cnn_att_20_05_24_5.keras')
        this.eval_cnn_att(model)
        probs = model.predict([first_partial_img_data_X,second_partial_img_data_X,third_partial_img_data_X,global_img_data_X])
        print(probs)
        predicted_class = np.argmax(probs)
        print("Predicted class", translate[int_class_to_animal[predicted_class]])
        # print("Predicted class index:", predicted_class)
        # #factorised_to_original = MyUtils.load_dict("balanced" + "_transformation")
        # print("Predicted class: ", predicted_class)
        print("****************************************************************")
       # model.summary()
        
        this.show_heatmap(img_arr1, img_arr2, img_arr3,img_nump,model)
        
    def eval_cnn_att(this,model):
        # less than ideal solution
        training_data_for_evals = keras.utils.image_dataset_from_directory(
        "./predict_all_images",
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

        img_numps = np.concatenate([x for x, y in training_data_for_evals], axis=0)

        labels = np.concatenate([y for x, y in training_data_for_evals], axis=0)
        print(labels) 

        count = 0
        for i in range(len(img_numps)):
            current_img_nump = img_numps[i]
            global_img_data_X.append(current_img_nump)
            global_img_data_y.append(labels[i])

        # https://www.geeksforgeeks.org/splitting-arrays-in-numpy/
            splits = np.array_split(current_img_nump, 3)
        # splits2 = np.array_split(splits[2], 3)
            first_partial_img_data_X.append(splits[0])
            first_partial_img_data_y.append(labels[i])
            second_partial_img_data_X.append(splits[1])
            second_partial_img_data_y.append(labels[i])
            third_partial_img_data_X.append(splits[2])
            third_partial_img_data_y.append(labels[i])

        global_img_data_X = np.array(global_img_data_X)
        #global_img_data_X = splits[2]
        global_img_data_y = np.array(global_img_data_y)
        first_partial_img_data_X = np.array(first_partial_img_data_X)
        first_partial_img_data_y = np.array(first_partial_img_data_y)
        second_partial_img_data_X = np.array(second_partial_img_data_X)
        second_partial_img_data_y = np.array(second_partial_img_data_y)
        third_partial_img_data_X = np.array(third_partial_img_data_X)
        third_partial_img_data_y = np.array(third_partial_img_data_y)

        img_arr = global_img_data_X[31]
        img_arr = img_arr.astype(np.uint8)
        image = Image.fromarray(img_arr)
        # image.show()
        print('global label',global_img_data_y[31])

        img_arr = first_partial_img_data_X[31]
        img_arr = img_arr.astype(np.uint8)
        image = Image.fromarray(img_arr)
        # image.show()
        print('first label',first_partial_img_data_y[31])

        img_arr = second_partial_img_data_X[31]
        img_arr = img_arr.astype(np.uint8)
        image = Image.fromarray(img_arr)
        # image.show()
        print('second label',second_partial_img_data_y[31])

        img_arr = third_partial_img_data_X[31]
        img_arr = img_arr.astype(np.uint8)
        image = Image.fromarray(img_arr)
        # image.show()
        print('third label',third_partial_img_data_y[31])

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

        X_testst = [X_test_top, X_test_mid, X_test_bot, X_test]
        factorised_to_original = []
        eval(X_testst,y_test,model,factorised_to_original)


        

        
    def show_heatmap(this,img_arr1, img_arr2, img_arr3,img_nump,model):
        # NOT ORIGINAL
        # # https://keras.io/examples/vision/grad_cam/
        img_arr1_expand = np.expand_dims(img_arr1,axis=0)
        img_arr2_expand = np.expand_dims(img_arr2,axis=0)
        img_arr3_expand = np.expand_dims(img_arr3,axis=0)
        img_nump_expand = np.expand_dims(img_nump,axis=0)
        imgs = [img_arr1_expand, img_arr2_expand, img_arr3_expand,img_nump_expand]
       
        # Make model
        model = model
        # Remove last layer's softmax
        model.layers[-1].activation = None

        # Generate class activation heatmap
        heatmap_glob = gc.make_gradcam_heatmap(imgs, model, "multi_head_attention_3")
        gc.save_and_display_gradcam(img_nump, heatmap_glob)



    # def runner(this):
    #     extractor = Predictions("predict")
        
        



#print(extractor.animal_stats)
load_and_copy()
cnn_predict = Predictions("predict")

cnn_predict.spec_run()
print("cnn", cnn_predict.animal_noise_data)
cnn_predict.predict_img()

extractor = Predictions("predict")
#extractor.record_and_save_noise()
extractor.process_sounds()

# print(extractor.animal_stats)
df = MyUtils.create_df(extractor.animal_stats)
print("cnn", cnn_predict.animal_noise_data)
MyUtils.data_frame_to_csv(df,"po.csv")  
extractor.predictions('balanced_concat.keras','balanced_single.keras','trimmed_concat.keras','trimmed_single.keras')

