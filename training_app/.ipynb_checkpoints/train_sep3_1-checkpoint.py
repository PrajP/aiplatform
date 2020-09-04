
# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys

import fire
import pickle
import numpy as np
import pandas as pd

import json

import pickle
import time


import hypertune

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn import linear_model
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from googleapiclient import discovery
from googleapiclient import errors




def train_evaluate(job_dir, training_dataset_path, validation_dataset_path, alpha, max_iter, hptune):
    
   
    #obj = open('../data/Anonymized_Fermentation_Data_final.xlsx', 'rb')
    #obj = open('/home/jupyter/aiplatform/data/Anonymized_Fermentation_Data_final.xlsx', 'rb')
    obj = INPUT_FILE
    #obj = 'gs://input_data_amy_bkt1/input_data/Anonymized_Fermentation_Data_final.xlsx'
    data = pd.read_excel(obj,sheet_name='data') 
    meta_data = pd.read_excel(obj,sheet_name='meta data') 
    
    
    if not hptune:
        df_train = pd.concat([df_train, df_validation])


### ==============================================================================####
    #data = pd.read_excel(bkt_excl,sheet_name='data') 
    df_train = pd.read_excel(training_dataset_path,sheet_name='data')
    df_validation = pd.read_excel(validation_dataset_path,sheet_name='data')
    meta_data = pd.read_excel(training_dataset_path,sheet_name='meta data') 
    
    #df_train = pd.read_csv(training_dataset_path)
    #df_validation = pd.read_csv(validation_dataset_path)

    if not hptune:
        df_train = pd.concat([df_train, df_validation])

    #numeric_feature_indexes = slice(0, 10)
    #categorical_feature_indexes = slice(10, 12)
   
    
    #Prepare data for analysis
    #Split out numeric from categorical varibles

    numeric_vars = ((df_train.dtypes == 'float64') | (df_train.dtypes == 'int64')) & (meta_data['variable type'] == 'independent').values
    numeric_x_data = df_train[df_train.columns[numeric_vars]]

    numeric_vars_val = ((df_validation.dtypes == 'float64') | (df_validation.dtypes == 'int64')) & (meta_data['variable type'] == 'independent').values
    numeric_x_data_val = df_validation[df_validation.columns[numeric_vars_val]]

    
    cat_vars = ((df_train.dtypes == 'string') | (df_train.dtypes == 'object')) & (meta_data['variable type'] == 'independent').values
    cat_x_data = df_train[df_train.columns[cat_vars]]

    #Things to try to predict

    y_data = df_train[df_train.columns[(meta_data['target'] == 1).values]]
    y_data_val = df_validation[df_validation.columns[(meta_data['target'] == 1).values]]
    
   
    # meta data about variables

    meta_data = meta_data.set_index('name')    

    #Impute missing with median #handle missing values with median. SKlearn provides imputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')

    #auto scale
    scaler = StandardScaler()

    #pipe = Pipeline([('imputer',imputer),('scaler', scaler)])
    
    #preprocessor = ColumnTransformer(
    #transformers=[
    #    ('num', StandardScaler(), numeric_x_data),
    #    ('cat', OneHotEncoder(), cat_x_data) 
    #])

    pipe = Pipeline([
        ('imputer',imputer), ('scaler', scaler) 
    ])

    
    scaled_numeric_x_data = pipe.fit_transform(numeric_x_data)
    scaled_numeric_x_data_val = pipe.fit_transform(numeric_x_data_val)
    
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_numeric_x_data)
    pca_result_val = pca.fit_transform(scaled_numeric_x_data_val)

    
    #pipe = Pipeline([
    #    ('preprocessor', preprocessor),('imputer',imputer),
    #    ('sgdregressor', SGDRegressor(loss='squared_loss',tol=1e-3))
    #])

    
    pipeline = Pipeline([
        ('sgdregressor', SGDRegressor(loss='squared_loss',tol=1e-3))
    ])
    
    
    ##########
    #preprocessor = ColumnTransformer(
    #transformers=[
    #    ('num', StandardScaler(), numeric_feature_indexes),
    #    ('cat', OneHotEncoder(), categorical_feature_indexes) 
    #])

    #pipeline = Pipeline([
    #    ('preprocessor', preprocessor),
    #    ('classifier', SGDClassifier(loss='log',tol=1e-3))
    #])

    #num_features_type_map = {feature: 'float64' for feature in df_train.columns[numeric_feature_indexes]}
    #df_train = df_train.astype(num_features_type_map)
    #df_validation = df_validation.astype(num_features_type_map) 
     
    
    print('Starting training: alpha={}, max_iter={}'.format(alpha, max_iter))
    df_train = get_results(pca_result,'pca-', add = y_data)
    df1_train = df_train
    df2_train = df1_train.dropna()
    
    df_validation = get_results(pca_result_val,'pca-', add = y_data_val)
    df1_validation = df_validation
    df2_validation = df1_validation.dropna()

    
    #X_train = df_train.drop("Run_Execution","Run_Performance","Product_Produced__g","Titer_End__g_over_kg", axis=1)
    #y_train = df_train["Product_Produced__g"]
    #X_train = df2_train.drop("Run_Execution","Run_Performance","Product_Produced__g","Titer_End__g_over_kg", axis=1)
    X_train = df2_train.drop(columns=["Run_Execution","Run_Performance","Product_Produced__g","Titer_End__g_over_kg"])
    y_train = df2_train["Product_Produced__g"]


    #pipeline.set_params(classifier__alpha=alpha, classifier__max_iter=max_iter)
    pipeline.set_params(sgdregressor__alpha=alpha, sgdregressor__max_iter=max_iter)   
    pipeline.fit(X_train, y_train)

    if hptune:
        X_validation = df2_validation.drop(columns=["Run_Execution","Run_Performance","Product_Produced__g","Titer_End__g_over_kg"])
        y_validation = df2_validation["Product_Produced__g"]
        accuracy = pipeline.score(X_validation, y_validation)
        print('Model accuracy: {}'.format(accuracy))
        # Log it with hypertune
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
          hyperparameter_metric_tag='accuracy',
          metric_value=accuracy
        )

    # Save the model
    if not hptune:
        model_filename = 'model.pkl'
        with open(model_filename, 'wb') as model_file:
            pickle.dump(pipeline, model_file)
        gcs_model_path = "{}/{}".format(job_dir, model_filename)
        subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path], stderr=sys.stdout)
        print("Saved model in: {}".format(gcs_model_path)) 

        

def get_results(res,prefix='',ncol=3, add=None):
    #collect results    
    out= pd.DataFrame()
    for i in range(ncol):
        key = prefix + str(i+1)
        value = res[:,i]
        out.loc[:,key] = value
    
    if add is not None:
        out = pd.concat([out,add],axis=1)
    
    return out

        
if __name__ == "__main__":
    fire.Fire(train_evaluate)



