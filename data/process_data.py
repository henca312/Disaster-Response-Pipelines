import sys
from sqlalchemy import create_engine
import pandas as pd 


def load_data(messages_filepath, categories_filepath):
    
    '''
    Load relevant data for analysis
    -> disaster_categories.csv
    -> disaster_messages.csv
    
    Input:
    -> Filepath for messages
    -> Filepath for categories
    
    Output:
    -> Merged Dataset between messages and categories
    
    '''
      
    
    messages = pd.read_csv(messages_filepath,sep=',')
    categories = pd.read_csv(categories_filepath, sep=',')
    
    
    # merge datasets
    df = pd.merge(categories,messages, on=['id'])
    
    return df


def clean_data(df):
    
    ''''
    Clean dataframe to provide the basis for analysis
    
    Input:
    -> dataframe that should be cleaned
    
    Output:
    -> Cleaned dataframe
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = df['categories'].iloc[0]
    
    def Convert(string): 
        li = list(string.split(";")) 
        return li 

    category_colnames = Convert(row)
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str)
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)


    # Keep Column Names
    categories.columns = category_colnames

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'],axis=1)
    
    
    # concatenate the original dataframe with the new `categories` dataframe

    frames = [df,categories]

    df = pd.concat(frames, axis=1,sort=False)
    
    #Drop duplicates
    df = df.drop_duplicates(keep=False)
    
    return df


def save_data(df, database_filename):
    
    ''''
    Input:
    -> dataframe that should be implemented in database
    
    Output:
    -> Table containing the data of the input dataframe in the sql database
    '''
    
    engine = create_engine('sqlite:///'+database_filename)
    
    df.to_sql('InsertTableName', engine, index=False,if_exists='replace')  
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()