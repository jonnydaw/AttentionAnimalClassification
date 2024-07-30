import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_classif


class DataFormating():
    def __init__(this):
        this.saved_model_type = None
        this.df = None
        this.y = None
        this.X = None
        this.X_train, this.X_test, this.y_train, this.y_test = None,None,None,None
        this.input_shape = None
        this.output_shape = None

    def fully_balanced(this,bool):
        '''
            Determining whether the dataset to be loaded is the fully balanced one
            or the trimmed one.
            Args:
                bool (boolean) : True for the fully balanced, False for the other
            Returns:
                None
            Raises:
                Not yet implemented
        '''

        if(bool):
            this.df = pd.read_csv("processed_sound_stats_balanced.csv")
            this.saved_model_type = "balanced"
        else:
            this.df = pd.read_csv("processed_sound_stats_trimmed.csv")
            print(this.df)
            this.saved_model_type = "trimmed"
    
    # def best_features_set_traing_and_test(this):
    #     # not used
    #     # https://stackoverflow.com/questions/39839112/the-easiest-way-for-getting-feature-names-after-running-selectkbest-in-scikit-le
    #     this.y = this.df.iloc[:, 0] 
    #     this.X = this.df[[x for x in this.df.columns[1:]]]
    #     KBest = SelectKBest(f_classif, k = int(len(this.X.columns) * 0.67)).fit(this.X, this.y)
    #     f = KBest.get_support(1)
    #     this.X = this.X[this.X.columns[f]]
    #     this.X_train, this.X_test, this.y_train, this.y_test = train_test_split(this.X,this.y, test_size = 0.3, random_state = 14)

    def set_training_and_test_data(this):
        this.y = this.df.iloc[:, 0] 
        this.X = this.df[[x for x in this.df.columns[1:]]]
        this.X_train, this.X_test, this.y_train, this.y_test = train_test_split(this.X,this.y, test_size = 0.3, random_state = 14)

    def set_input_shape(this):
        this.input_shape = (len(this.X.columns),)

    def set_output_shape(this):
        this.output_shape = len(set(this.y))
        print(this.output_shape)
# https://www.geeksforgeeks.org/abstract-classes-in-python/
class AbstractTunedModel(ABC,DataFormating):
    def __init__(this):
        this.tuner = None
        this.es = None
        this.history = None
        this.model = None

    @abstractmethod
    def splitter(this):
        pass

    @abstractmethod
    def model_builder(this, hp):
        '''
            Outliniing the potential different hyperparameters the model may have.
            Args:
                hp (???) : ???
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        raise NotImplementedError

    @abstractmethod
    def tune(this):
        '''
            Using random search in order build and search the models
            for the best hps.
            Args:
                hp (???) : ???
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        raise NotImplementedError



    def early_stop(this):
        '''
            Defining the criteria for early stopping
            Args:
                None
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        this.es = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='auto',
            patience=2,
            restore_best_weights = True
        )

    def search(this):
        '''
            Search to find the best hps
            Args:
                None
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        this.tuner.search(this.X_train, this.y_train, epochs=250,
                          validation_split=0.5,
                          callbacks=[this.es])

    def fit_model(this):
        '''
            Fitting the model based on the best hps
            Args:
                None
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        hyper_parameters = this.tuner.get_best_hyperparameters(num_trials=1)[0]
        this.model = this.tuner.hypermodel.build(hyper_parameters)
        this.history = this.model.fit(this.X_train, this.y_train, epochs=150,
                            validation_split=0.5,
                            shuffle=True,
                            callbacks=[this.es]
                            )
    
    def get_model(this):
        return this.model
        
        
    def stats(this):
        '''
            Plotting the validation and training acc and loss.
            Args:
                name (str) : file name for the model
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        history_dict = this.history.history
        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs = range(1, len(acc) + 1)

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

    def run(this):
        ## runner
        this.set_training_and_test_data()
        this.set_input_shape()
        this.set_output_shape()
        this.splitter()
        this.tune()
        this.early_stop()
        this.search()
        this.fit_model()
        this.stats()

    # def run_best_features(this):
    #     ## runner
    #     #this.load_df()
    #     this.best_features_set_traing_and_test()
    #     this.set_input_shape()
    #     this.set_output_shape()
    #     this.splitter()
    #     this.tune()
    #     this.early_stop()
    #     this.search()
    #     this.fit_model()
    #     # this.save_model()
    #     this.stats()





