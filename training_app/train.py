
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
from sklearn.model_selection import train_test_split  ## from analysis

#simple sklearn impute and scale numeric pipeline
from sklearn.pipeline import Pipeline ## from analysis
from sklearn.impute import SimpleImputer ## from analysis
from sklearn.preprocessing import StandardScaler ## from analysis
import numpy as np ## from analysis

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV


import functools


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




def train_evaluate(job_dir, input_file, training_dataset, validation_dataset, n_estimators, max_depth, min_samples_leaf, hptune):
    
   
    obj = input_file
    print("obj", obj)
    data = pd.read_excel(obj,sheet_name='data') 
    print("Opened excel file and assigned to data")
    meta_data = pd.read_excel(obj,sheet_name='meta data') 
    print("Just after file opening")
    
    if not hptune:
        df_train = pd.concat([df_train, df_validation])


    ## Preprocess    
    #Prepare data for analysis
    #Split out numeric from categorical varibles

    ##var_type_filter = [x in ['physiological','biochemical','process'] for x in meta_data['variable type']]
    var_type_filter = [x in ['independent'] for x in meta_data['variable type']]
    var_dtype_filter = (data.dtypes == 'float64') | (data.dtypes == 'int64')

    numeric_vars = (var_type_filter & var_dtype_filter).values
    numeric_x_data = data[data.columns[numeric_vars]]

    #things to try to predict
    y_data = data[data.columns[(meta_data['target'] == 1).values]]

    #meta data about variables
    meta_data = meta_data.query('name in {}'.format(list(data.columns[numeric_vars].values))).set_index('name')
 
    #Variables which will be used to build the model
    ####data.columns[numeric_vars].values
        


    model_target = 'Run_Performance' ## Select target for classification
    
    print("before splitting data")
    #maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(numeric_x_data, y_data, test_size=0.25, stratify = y_data[model_target], random_state=42)

    #split train set to create a pseudo test or validation dataset
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.33, stratify= y_train[model_target], random_state=42)
    
    
    print('The training, validation and test data contain {}, {} and {} rows respectively'.format(len(X_train),len(X_validate),len(X_test)))

    #save_list_train = ['X_train','y_train','meta_data']
    #save_list_validate = ['X_validate','y_validate','meta_data']
    #save_list_test = ['X_test','y_test','meta_data']

    #for x in save_list_train:
    #    print("training_dataset", training_dataset)
    #    obj = pd.DataFrame(globals()[x])
    #    #cmd = "obj.to_csv('../data/{}.csv')".format(x)
    #    #cmd = "obj.to_csv('/home/jupyter/aiplatform/data/{}.csv')".format(x)
    #    cmd = "obj.to_excel('{}/{}.xlsx')".format(training_dataset, x)
    #    eval(cmd)
    
    #for x in save_list_validate:
    #    print("validation_dataset", validation_dataset)
    #    obj = pd.DataFrame(globals()[x])
    #    #cmd = "obj.to_csv('../data/{}.csv')".format(x)
    #    #cmd = "obj.to_csv('/home/jupyter/aiplatform/data/{}.csv')".format(x)
    #    cmd = "obj.to_excel('{}/{}.xlsx')".format(validation_dataset, x)
    #    eval(cmd)    
    
    #for x in save_list_test:
    #    print("testing_dataset", testing_dataset)
    #    obj = pd.DataFrame(globals()[x])
    #    #cmd = "obj.to_csv('../data/{}.csv')".format(x)
    #    #cmd = "obj.to_csv('/home/jupyter/aiplatform/data/{}.csv')".format(x)
    #    cmd = "obj.to_excel('{}/{}.xlsx')".format(testing_dataset, x)
    #    eval(cmd)    
    

    ## Train, optimize and validate predictive model
    ### Train



    classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf

     )

    #numeric_transformer = Pipeline([
    #  ('imputer', SimpleImputer(strategy='median')),
    #  ('scaler', StandardScaler()),
    # ])


    #transform_list = []
      # If there exist numerical columns
    #transform_list.extend([
    #    ('numeric', numeric_transformer)
    #])

    #preprocessor = ColumnTransformer(transform_list)
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    #auto scale
    scaler = StandardScaler()

    estimator = Pipeline([
      ('imputer', imputer),
      ('scaler', scaler),
      ('classifier', classifier),
    ])
    
    
    
    #estimator = Pipeline([
      #('preprocessor', preprocessor),
    
     # ('classifier', classifier),
    #])

    #prepare data for modeling
    #use the pipeline created above
    #_X_train = pipe.fit_transform(X_train)
    #_y_train = y_train[model_target]    ## selected target label for prediction
    #_X_test = pipe.fit_transform(X_validate)
    #_y_test = y_validate[model_target]

    #_X_train = pipe.fit_transform(X_train)
    _X_train = X_train    
    _y_train = y_train[model_target]    ## selected target label for prediction
    _X_test = X_validate
    _y_test = y_validate[model_target]
    

    print('Starting training: alpha={}, max_iter={}'.format(n_estimators, max_depth, min_samples_leaf))

    #estimator.set_params(classifier__alpha=alpha, classifier__max_iter=max_iter) 
    estimator.set_params(classifier__n_estimators=n_estimators, classifier__max_depth=max_depth, classifier__min_samples_leaf=min_samples_leaf) 
    #pipeline.fit(X_train, y_train)
    estimator.fit(_X_train, _y_train)

    
    if hptune:
        #X_validation = df2_validation.drop(columns=["Run_Execution","Run_Performance","Product_Produced__g","Titer_End__g_over_kg"])
        #y_validation = df2_validation["Product_Produced__g"]
        accuracy = estimator.score(_X_test, _y_test)
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
            pickle.dump(estimator, model_file)
        gcs_model_path = "{}/{}".format(job_dir, model_filename)
        subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path], stderr=sys.stdout)
        print("Saved model in: {}".format(gcs_model_path)) 

    

        
if __name__ == "__main__":
    fire.Fire(train_evaluate)



