from keras.models import Sequential, Model
import keras
from keras.layers import Dense
from keras import regularizers
import keras_tuner as kt
from keras.layers import BatchNormalization, Concatenate,Dropout
from keras.layers import Flatten
import tensorflow
from abstract_tuned_models import AbstractTunedModel

class ConcatDense(AbstractTunedModel):
    def __init__(this):
         super().__init__()

    def model_builder(this, hp):
        '''
            Outliniing the potential different hyperparameters the model may have.
            Here there are actually two different neural networks which are subsequently 
            concatenated.
            Args:
                hp (???) : ???
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        hp_activation_branch_1 = hp.Choice('activation_branch_1', values = ['LeakyReLU','gelu','relu']) 
        hp_layer_1_branch_1 = hp.Int('layer_1_branch_1', min_value = 8, max_value = 512, step = 16)
        hp_layer_2_branch_1 = hp.Int('layer_2_branch_1', min_value = 8, max_value = 512, step = 16)

        hp_activation_branch_2 = hp.Choice('activation_branch_2', values = ['tanh','selu','elu']) 
        hp_layer_1_branch_2 = hp.Int('layer_1_branch_2', min_value = 8, max_value = 512, step = 8)
        hp_layer_2_branch_2 = hp.Int('layer_2_branch_2', min_value = 8, max_value = 512, step = 8)
        #hp_layer_3_branch_2 = hp.Int('layer_3_branch_2', min_value = 512, max_value = 1024, step = 8)

        hp_learning_rate = hp.Choice('learning_rate', values = [0.01, 0.001, 0.0001,0.00005])

        branch1 = Sequential()
        branch1.add(Flatten(input_shape = this.input_shape))
        branch1.add(BatchNormalization())
        branch1.add(Dense(units = hp_layer_1_branch_1, activation = hp_activation_branch_1))    
        branch1.add(Dense(units = hp_layer_2_branch_1, activation = hp_activation_branch_1, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)))
        branch1.add(Dropout(0.3))

        branch2 = Sequential()
        branch2.add(Flatten(input_shape = this.input_shape))
        branch2.add(BatchNormalization())
        branch2.add(Dense(units = hp_layer_1_branch_2, activation = hp_activation_branch_2))    
        branch2.add(Dense(units = hp_layer_2_branch_2, activation = hp_activation_branch_2, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)))
        branch2.add(Dense(units = hp_layer_2_branch_2, activation = hp_activation_branch_2, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)))
        branch2.add(Dropout(0.1))

        concatenated = Concatenate()([branch1.output, branch2.output])
        output_layer = Dense(this.output_shape, activation='softmax')(concatenated)  
        model = Model(inputs=[branch1.input, branch2.input], outputs=output_layer)
        model.compile(optimizer = tensorflow.keras.optimizers.Nadam(learning_rate = hp_learning_rate),
                            loss='sparse_categorical_crossentropy', 
                            metrics = ['accuracy'] )
        return model
    
    def splitter(this):
        '''
            Re-structuring the data so it is in a suitable format 
            for the two nns
            Args:
                hp (???) : ???
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        this.X_train = [this.X_train,this.X_train]
        print(this.X_train[1])

    def tune(this):
        this.tuner = kt.RandomSearch(
        this.model_builder,
        objective = 'val_accuracy',
        max_trials=15,
        directory='dir',
        overwrite=True, # reloading prev model hps when false
        project_name = 'concat_test' + this.saved_model_type
        )


if __name__ == "__main__":
    cd_balanced = ConcatDense()
    cd_balanced.fully_balanced(True)
    cd_balanced.run()
    cd_balanced.save_model(cd_balanced.saved_model_type + "_concat.keras")

    cd_trimmed = ConcatDense()
    cd_trimmed.fully_balanced(False)
    cd_trimmed.run()
    cd_trimmed.save_model(cd_trimmed.saved_model_type + '_concat.keras')
