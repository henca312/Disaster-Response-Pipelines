# import libraries

import sys
from sqlalchemy import create_engine
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import time

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer


from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV , ShuffleSplit


from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from statistics import mean 
import statistics
import pickle


def load_data(database_filepath, table_name = 'InsertTableName'):
    
    '''
    Load Data from sql database
    
    Input:
        -> databasse_filepath: containing the path to the database
        
    Output:
        -> X: Containing the features for the machine learning model
        -> Y: Containg the values that should be predicted for the machien learning model
    '''
    
    # Load data from database
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(table_name, engine)
    
    X=df['message']
    Y=df.drop(['id','message','original','genre'],axis=1)
    category_names = Y.columns
    
    # Convert to dataframe to make sure that the data is not being formated as a series
    #Y = pd.DataFrame(Y) 
    
    # Convert all columns to integer values
    #for col in Y.columns:
    #    Y[col]=Y[col].astype(str).astype(int)
    

    return X,Y, category_names

def tokenize(text):
    
    '''
    Tokenize data
    
    Input:
        -> text data
    Output:
        -> Cleaned tokens
    '''
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())    
    # tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    ## lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    tokens_cleaned=[]
    for element in tokens:
        cleaned_element=lemmatizer.lemmatize(element)
        tokens_cleaned.append(cleaned_element)
        
    tokens = tokens_cleaned
    
    return tokens      
   

def build_model():
    
    '''
    Build model for Machine Learning
    -> Using the best parameters 
    '''
    
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize,lowercase=False)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs=-1))) 
        ])
    
    # Defining Parameters for Grid Search
    param_grid = { 
           "clf__estimator__criterion": ["gini"],
           'clf__estimator__max_depth': [50],
           "clf__estimator__max_features" : ["auto"],
           "clf__estimator__min_samples_split": [8]
            }
    
    # calculate Grid
    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, n_jobs=-1, cv=3, verbose=10)

    
    return grid


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Evaluate machine learning model
    
    '''
    
    # Predict Y Values based on the model
    Y_pred = model.predict(X_test)

    
        # Key Performance Indicators to describe the quality of the prediction
    KPI_accuracy_list = []
    KPI_precision_list = []
    KPI_recall_list = []
    KPI_f1_list_list = []
    
    # Iteration over metrics_label to calculate the KPI
    for element in range(len(metrics_label)):
        KPI_accuracy = accuracy_score(y_test[metrics_label[element]].values,Y_pred[:,element])
        KPI_precision = precision_score(y_test[metrics_label[element]].values,Y_pred[:,element],average='weighted')
        KPI_recall = recall_score(y_test[metrics_label[element]].values,Y_pred[:,element],average='weighted')
        KPI_f1_list = f1_score(y_test[metrics_label[element]].values,Y_pred[:,element],average='weighted')

 
    
        # Append Results to the KPI LIST
        KPI_accuracy_list.append(KPI_accuracy)
        KPI_precision_list.append(KPI_precision)
        KPI_recall_list.append(KPI_recall)
        KPI_f1_list_list.append(KPI_f1_list)
    
    
    # Dictionary 
    Results = {'Metrics':metrics_label,'Accuracy KPI':KPI_accuracy_list, 'Precision KPI':KPI_precision_list, 
              'Recall KPI':KPI_recall_list,'F1 KPI':KPI_f1_list_list}  
    
                  
    # Dataframe to store results
    df_results = pd.DataFrame.from_dict(Results)

    
    # Calculate Mean Value for each KPI
    Accurarcy_MEAN = statistics.mean(KPI_accuracy_list)
    Precision_MEAN = statistics.mean(KPI_precision_list)
    Recall_MEAN = statistics.mean(KPI_recall_list)
    F1_MEAN = statistics.mean(KPI_f1_list_list)
    
    # Print Values
    print("Mean Accuracy:",Accurarcy_MEAN,)
    print("Mean Precision:",Precision_MEAN,)
    print("Mean Recall:",Recall_MEAN,)
    print("Mean F1:",F1_MEAN,)

        
    return df_results
   
    

def save_model(model, model_filepath):
    '''
    SAVE Machline Learning model as pickle file
    '''
    with open(model_filepath, 'wb') as file:
     pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')       
               
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()