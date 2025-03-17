import pandas as pd

from src.recommender_systems.baseline import BaselineMostClicked
from src.recommender_systems.collaborative_filtering import ALSMatrixFactorization
from src.data_normalization import data_normalization


def compare_baseline_als():
    """ Compare the baseline and ALS matrix factorization models. """
    # Load the data
    data, _ = data_normalization(validation=False, try_load=True)
    
    # Initialize the models
    baseline = BaselineMostClicked()
    als = ALSMatrixFactorization()
    
    # Split the data
    data['behaviors'] = data['behaviors'].sort_values('Time')
    split_index = int(0.8 * len(data['behaviors']))
    data_train = {
        'behaviors': data['behaviors'].iloc[:split_index],
        'Impressions': data['Impressions'],
        'news': data['news'],
        'History': data['History'],
        'Title entities': data['Title entities'],
        'Abstract entities': data['Abstract entities'],
    }
    data_test = {
        'behaviors': data['behaviors'].iloc[split_index:],
        'Impressions': data['Impressions'],
        'news': data['news'],
        'History': data['History'],
        'Title entities': data['Title entities'],
        'Abstract entities': data['Abstract entities'],
    }
    
    # Fit the models
    baseline.fit(data_train)
    als.fit(data_train)
    
    # Predict
    user_id = 'U13740'
    time = data_test['behaviors']['Time'].iloc[0]
    k = 10
    baseline_recommendations = baseline.predict(user_id, time, k)
    als_recommendations = als.predict(user_id, time, k)
    
    # Compare
    