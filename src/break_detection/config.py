MODEL_PARAMS = {
    'xgb': {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse', 
        'max_depth': 6,
        'learning_rate': 0.3,
        'n_estimators': 100
    },
    'rf': {
        'n_estimators': 100,
        'random_state': 1
    },
    'mlp': {
        'hidden_layer_sizes': (55, 34, 13),
        'random_state': 1,
        'max_iter': 500
    }
}

FEATURE_CONFIG = {
    'window_size': 3,
    'outlier_threshold': 3,
    'adaptive_window_min': 10
}