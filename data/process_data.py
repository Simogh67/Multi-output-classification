import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load and merge messages and categories datasets
    
    arguments:
    messages_filepath: string. filepath for messages dataset.
    categories_filepath: string. filepath for categories dataset.
       
    outputs:
    df: dataframe. the dataframe contains the messages and categories datasets.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, how = 'left',on = 'id')
    return df


def clean_data(df):
    """This function cleans the dataframe by removing duplicates and converting categorical columns to binary values
    
    Args:
    df: dataframe. the dataframe contains the messages and categories datasets.
       
    Returns:
    df: dataframe. a clean version of the input 
    """
    # create a dataframe of the individual category columns
    categories= df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # extracting a list of new column names for categories
    category_colnames = row.map(lambda x: x[:-2])
    categories.columns = category_colnames
    # Converting category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].map(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(str)
        categories[column] = pd.to_numeric(categories[column])
        # drop the original categories column from `df`
    df=df.drop(['categories'], axis=1)
    # Concatenate the dataframe with the new `categories` 
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df=df.drop_duplicates()
    # Remove rows whose value equals 2 from the dataframe
    df = df[df['related'] != 2]
    
    return df


def save_data(df, database_filename):
    """ This function saves the cleaned dataframe to SQLite database by taking the name of the database and the cleaned dataframe"""
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Disaster_data', engine, index=False)  


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
