from keras.models import Sequential
import keras
from keras.layers import Dense
from keras import regularizers
import keras_tuner as kt
from keras.layers import BatchNormalization
from keras.layers import Flatten
import tensorflow
from abstract_tuned_models import AbstractTunedModel
import pandas as pd
import numpy as np


class SingleDense(AbstractTunedModel):
    def __init__(this):
         super().__init__()

   
        
    def model_builder(this, hp):
        '''
            Outliniing the potential different hyperparameters the model may have.
            The model being defined here is a regular feed-forward nn.
            Args:
                hp (???) : ???
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        hp_activation = hp.Choice('activation', values = ['LeakyReLU','gelu','relu']) 
        hp_layer_1 = hp.Int('layer_1', min_value = 0, max_value = 512, step = 16)
        hp_layer_2 = hp.Int('layer_2', min_value = 0, max_value = 512, step = 16)
        hp_learning_rate = hp.Choice('learning_rate', values = [0.01, 0.001, 0.0001,0.00005])
        model = Sequential()
        model.add(Flatten(input_shape = this.input_shape))
        model.add(BatchNormalization())
        model.add(Dense(units = hp_layer_1, activation = hp_activation))   
        # https://keras.io/api/layers/regularizers/
        model.add(Dense(units = hp_layer_2, activation = hp_activation, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)))
        # 6, 10
        model.add(Dense(this.output_shape, activation = 'softmax'))
        # prepending tensorflow makes a big difference when reloading
        model.compile(optimizer = tensorflow.keras.optimizers.Nadam(learning_rate = hp_learning_rate),
                            loss='sparse_categorical_crossentropy', 
                            metrics = ['accuracy'] )
        return model
    
    def splitter(this):
        # ignored here
        return super().splitter()

    def tune(this): 
        this.tuner = kt.RandomSearch(
        this.model_builder,
        objective = 'val_accuracy',
        max_trials=15,
        directory='dir',
        overwrite=True,
        project_name= 'single_attention' + this.saved_model_type
        )
    


   


if __name__ == "__main__":
    sd_balanced = SingleDense()
    sd_balanced.fully_balanced(True)
    sd_balanced.run()
    sd_balanced.save_model(sd_balanced.saved_model_type +"_single.keras")
    
    sd_trimmed = SingleDense()
    sd_trimmed.fully_balanced(False)
    sd_trimmed.run()
    sd_trimmed.save_model(sd_trimmed.saved_model_type +"_single.keras")
    # sd.save_model()




    # def save_model(this):
    #     hyper_parameters = this.tuner.get_best_hyperparameters(num_trials=1)[0]
    #     s_model = this.tuner.hypermodel.build(hyper_parameters)
    #     s_model.save("single_dense_1.keras")

    # def scale(this):
    #     df_prediction = pd.read_csv("po.csv")
    #     scale_template = pd.read_csv("scale_template.csv")
    #     scale_template = scale_template.drop("species",axis=1)      
    #     df_prediction = df_prediction.drop("species",axis=1)
    #     print(df_prediction)     
    #     max_vals = scale_template.max(axis=0).values
    #     min_vals = scale_template.min(axis=0).values
    #     float_value_list = [max_vals.tolist()[0]]
    #     print(df_prediction.values.tolist()[0])
    #     print(float_value_list)
    #     scaled_vals = (df_prediction.values - min_vals) / (max_vals - min_vals)
    #     return scaled_vals
    
    # def predict(this):
    #     x = this.scale()
    #     print(x)
    #     x = this.model.predict(x)
    #     print(x)
    #     predicted_class = np.argmax(x)
    #     print("Predicted class index:", predicted_class)