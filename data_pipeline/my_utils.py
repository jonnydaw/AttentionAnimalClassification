import pandas as pd
import pickle
class MyUtils:
    '''
        Collection of useful methods in an attempt to reduce 
        repetition.
    '''
    
    def data_frame_to_csv(df, filename):
        '''
            Converting df to csv
            Args:
                df (DataFrame) : dataframe to be converted
                filename (str) : name of file to be saved as csv. .csv needs to be appended
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        df.to_csv(filename, encoding='utf-8', index=False)

    def create_df(dict):
        '''
            Converting dictionary to df
            Args:
                dict (dict) : dictionary to be converted
            Returns:
                df (DataFrame) : dataframe populated by dictionary
            Raises:
                Not yet implemented
        '''
        # https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
        df = pd.DataFrame.from_dict(dict, orient='index')
        df = df.apply(pd.Series.explode)
        df['species'] = df.index
        columns = df.columns[::-1]
        df = df[columns]
        return df 
    
    def dump_dict(hasmap,file_name):
        '''
            Saving a dictionary using pickle
            Args:
                hasmap (dict) : dictionary to be save
                file_name (str) : file name of dumped dictionary. Appending
                                  .pickle is not necessary.  
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        #https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object
        with open(file_name + '.pickle', 'wb') as handle:
            pickle.dump(hasmap, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_dict(file_name):
        '''
            Loading a dictionary using pickle
            Args:
                file_name (str) : file name of dumped dictionary. Appending
                                  .pickle is not necessary.  
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        with open(file_name + '.pickle', 'rb') as handle:
            gherkin = pickle.load(handle)
        return gherkin
    
    


