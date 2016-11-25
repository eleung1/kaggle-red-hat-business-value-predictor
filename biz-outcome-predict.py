# -*- coding: utf-8 -*-
"""
@author: Eric Leung

Red Hat Business Value Predictor.
"""

import pandas as pd
import numpy as np

# Examine people.csv

df_people = pd.read_csv('./data/people.csv', header=0)
df_people.head(5)

df_people.dtypes

# No missing values.  189118 rows.
df_people.info()

# quick summary
df_people.describe()

# Examine training activities

df_act_train = pd.read_csv('./data/act_train.csv', header=0)
df_act_train.head(5)

df_act_train.dtypes

df_act_train.info()

df_act_train.describe()

df_act_train['char_1'].unique()

df_act_train['activity_category'].unique()

df_act_train['char_2'].unique()

df_act_train['char_3'].unique()

df_act_train['char_4'].unique()

df_act_train['char_5'].unique()

df_act_train['char_6'].unique()

df_act_train['char_7'].unique()

df_act_train['char_8'].unique()

df_act_train['char_9'].unique()

df_act_train['char_10'].unique()

df_act_train['outcome'].unique()

# Check positive and negative class size for imbalance

df_act_train.size

negative_size = df_act_train[ df_act_train['outcome'] == 0 ].size
negative_size

positive_size = df_act_train[ df_act_train['outcome'] == 1 ].size
positive_size

print 'Negaitve %=' + str(float(negative_size)/float(df_act_train.size))
print 'Positive %=' + str(float(positive_size)/float(df_act_train.size))

print 'Negative class is more than positive class by ' + str(negative_size - positive_size)

# Preprocess data

# helper method that get last digit from a string
import re
def getLastDigitsFromString(x):
    if x != x:
        return -9999
    
    result = re.match('.*?([0-9]+)$', x)
    if result is None:
        return -9999
    else:
        return int(result.group(1))


df_act_train['people_id'] = df_act_train['people_id'].map( getLastDigitsFromString )
df_act_train['activity_id_num'] = df_act_train['activity_id'].map( getLastDigitsFromString )
df_act_train['activity_category'] = df_act_train['activity_category'].map( getLastDigitsFromString )
df_act_train['char_1'] = df_act_train['char_1'].map( getLastDigitsFromString )
df_act_train['char_2'] = df_act_train['char_2'].map( getLastDigitsFromString )
df_act_train['char_3'] = df_act_train['char_3'].map( getLastDigitsFromString )
df_act_train['char_4'] = df_act_train['char_4'].map( getLastDigitsFromString )
df_act_train['char_5'] = df_act_train['char_5'].map( getLastDigitsFromString )
df_act_train['char_6'] = df_act_train['char_6'].map( getLastDigitsFromString )
df_act_train['char_7'] = df_act_train['char_7'].map( getLastDigitsFromString )
df_act_train['char_8'] = df_act_train['char_8'].map( getLastDigitsFromString )
df_act_train['char_9'] = df_act_train['char_9'].map( getLastDigitsFromString )
df_act_train['char_10'] = df_act_train['char_10'].map( getLastDigitsFromString )

df_act_train.head(20)

# Train model

import graphlab as gl

# convert dataframe to sframe
sf_act_train = gl.SFrame(df_act_train)
sf_act_train.print_rows(num_rows=50)

model = gl.random_forest_classifier.create(sf_act_train, target='outcome',
                                           features=['people_id',
                                                     'activity_id_num',
                                                     'activity_category',
                                                     'char_1',
                                                     'char_2',
                                                     'char_3',
                                                     'char_4',
                                                     'char_5',
                                                     'char_6',
                                                     'char_7',
                                                     'char_8',
                                                     'char_9',
                                                     'char_10'],
                                           max_iterations=1, max_depth = 300)


# Use model to predict test/unseen data

df_act_test = pd.read_csv('./data/act_test.csv', header=0)
df_act_test['people_id'] = df_act_test['people_id'].map( getLastDigitsFromString )
df_act_test['activity_id_num'] = df_act_test['activity_id'].map( getLastDigitsFromString )
df_act_test['activity_category'] = df_act_test['activity_category'].map( getLastDigitsFromString )
df_act_test['char_1'] = df_act_test['char_1'].map( getLastDigitsFromString )
df_act_test['char_2'] = df_act_test['char_2'].map( getLastDigitsFromString )
df_act_test['char_3'] = df_act_test['char_3'].map( getLastDigitsFromString )
df_act_test['char_4'] = df_act_test['char_4'].map( getLastDigitsFromString )
df_act_test['char_5'] = df_act_test['char_5'].map( getLastDigitsFromString )
df_act_test['char_6'] = df_act_test['char_6'].map( getLastDigitsFromString )
df_act_test['char_7'] = df_act_test['char_7'].map( getLastDigitsFromString )
df_act_test['char_8'] = df_act_test['char_8'].map( getLastDigitsFromString )
df_act_test['char_9'] = df_act_test['char_9'].map( getLastDigitsFromString )
df_act_test['char_10'] = df_act_test['char_10'].map( getLastDigitsFromString )


sf_act_test = gl.SFrame(df_act_test)

#SArray
final_pred = model.predict(sf_act_test)
print(final_pred)

sf_act_test.add_column(final_pred, name='outcome')

# create final results
df_result = sf_act_test.to_dataframe()
df_result = df_result[ ['activity_id', 'outcome'] ]
print 'Result size=' + str(df_result.size)
print(df_result.head(5))
np.savetxt('outcome_prediction.csv',df_result, header='activity_id,outcome', fmt='%s,%i')