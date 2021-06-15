def get_mean(dataset='activitynet', norm_value=1):
    assert dataset in ['activitynet', 'kinetics', 'sports1m']
    print('Mean dataset: %s' % dataset)
    if dataset == 'activitynet':
        mean = [114.7748, 107.7354, 99.4750] 
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        mean = [110.63666788, 103.16065604, 96.29023126]
    elif dataset == 'sports1m':
        mean = [101.41, 97.66, 90.25]
    return [item / norm_value for item in mean]
