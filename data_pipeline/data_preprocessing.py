import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from my_utils import MyUtils
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

class DataPreprocessing:
    '''
    Class used in order to 'clean' the data before being ingested
    by a machine learning algorithm.
    '''
    def __init__(this):
        this.df = None
        this.df_scaled = None
    
    def load_csv(this,csv_location):
        """
            Loads dataframe from specified location

            Args:
                csv_location (str) : location of csv to convert to df

            Returns:
                None

            Raises:
                Not yet implemented
        """
        this.df = pd.read_csv(csv_location)


    #https://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
    def filter_rows_by_values(this, df, col, values):
        """
            Filters/ cuts out the values passed from the df passed

            Args:
                df (DataFrame) : The dataframe from which the value/s should be dropped
                col (str) : The column in the dataframe to target
                values (list) : Values in column. If there is a match this row will be deleted

            Returns:
                None

            Raises:
                Not yet implemented
        """
        this.df = df[~df[col].isin(values)]

    def balance_all(this):
        '''
            Balances the df e.g. if there are 5 species with a varying number of
            occurences then the dataframe will be adjusted so that all species appear
            at the same frequency
            Args:
                None
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        df_for_training_grouped = this.df.groupby("species")
        df_for_training_grouped.groups.values()
        #https://stackoverflow.com/questions/45839316/pandas-balancing-data
        frames_of_groups = [x.sample(df_for_training_grouped.size().min()) for y, x in df_for_training_grouped]
        this.df = pd.concat(frames_of_groups)

    # def keep_best_attributes(this):
    #     X_new = SelectKBest(f_classif, k=).fit_transform(X, y)

    
    def removeOutliers(this,characteristic,csv_name):
        '''
           Replaces values beyond one sd of the mean with median value of that characteristic/ feature 
           for that species

            Args:
                characteristic (str) : column name
                csv_name (str) : name of the new .csv file created  
            Returns:
                None
            Raises:
                Not yet implemented
        '''
         
        medians = this.df.groupby('species')[characteristic].transform('median')
       # https://stackoverflow.com/questions/72787698/pandas-finding-and-replacing-outliers-based-on-a-group-of-two-columns
        this.df[characteristic] = this.df[characteristic].mask(lambda s: (s - s.mean()).abs() > s.std(), medians)
        # this is saved in order to for data later on to be scaled 
        MyUtils.data_frame_to_csv(this.df, csv_name + ".csv")

    def scale(this,f_name):
        '''
           Scales the dataframe using MinMaxScaler().
           In additon to this it also factorises the species column
           e.g. Tavuk -> 7
           This factorisation is then saved for later use by calling 
           save_factorisation()
           
            Args:
                f_name (str) : filename where the factorisation 
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        columns_to_scale = this.df.columns[1:]
        scaler = MinMaxScaler()
        this.df_scaled = this.df.copy()
        this.df_scaled[columns_to_scale] = scaler.fit_transform(this.df_scaled[columns_to_scale])
        #https://stackoverflow.com/questions/46134201/how-to-get-original-values-after-using-factorize-in-python
        values, uniques = pd.factorize(this.df_scaled['species']) 
        this.df_scaled['species'] = values
        #https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set
        this.save_factorisation(uniques.tolist(),list(dict.fromkeys(values)),f_name)

    def save_factorisation(this,original,new,filename):
        '''
           Maps the original species value to its new factorised value.
           This is then saved.
           
            Args:
                original (list) : The original values from the df as a last.
                                  in this case the species.
                new (list) : The factorised values.
                filename (str) : name of the file where this will saved
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        factorised_to_original = {}
        print(original)
        print(new)
        for i in range(len(new)):
            factorised_to_original[new[i]] = original[i]
        MyUtils.dump_dict(factorised_to_original,filename)
        print(factorised_to_original)

    # def shuffle(this):
        
    #     this.df = shuffle(this.df)

    def df_to_csv(this,csv_name):
        '''
           Saves the scaled df to a csv.
           
            Args:
                csv_name (str) : name for csv file to be saved. .csv must be appended
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        return MyUtils.data_frame_to_csv(this.df_scaled, csv_name)

    def run(this):
        '''
            Runner.
            In this case it produces a csv for the trimmed
            version. Trimmed version being the version where species with a very low frequency
            are dropped as opposed to balancing
        '''
        this.load_csv("raw_sound_stats.csv")
        this.filter_rows_by_values(this.df,'species',['Esek','Maymun','Kedi-Part2','Kopek-Part2','Kus-Part2','Tavuk','Kurbaga'])
        this.removeOutliers([x for x in this.df.columns[1:]],"trimmed_template")
        this.scale("trimmed_transformation")
        this.df_to_csv("processed_sound_stats_trimmed.csv")

    def run_fully_balance(this):
        '''
        Runner
            In this case it produces a csv for the balanced version.
        '''
        this.load_csv("raw_sound_stats.csv")
        # These have been removed as they are either duplicates or still too few values
        this.filter_rows_by_values(this.df,'species',['Kedi-Part2','Kopek-Part2','Kus-Part2','Maymun','Esek'])
        this.balance_all()
        this.removeOutliers([x for x in this.df.columns[1:]],"balanced_template")
        this.scale("balanced_transformation")
        this.df_to_csv("processed_sound_stats_balanced.csv")
    


data_pre_processing = DataPreprocessing()
# data_pre_processing.run()
# data_pre_processing.run_fully_balance()

