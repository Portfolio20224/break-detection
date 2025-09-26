from sklearn.base import BaseEstimator, TransformerMixin
from break_detection import OutlierDetectionTransformer, TimeSeriesModel

class FeatureEngineeringStep(BaseEstimator, TransformerMixin):
    def __init__(self, FEATURE_CONFIG, MODEL_PARAMS):
        self.window_size = FEATURE_CONFIG['window_size']
        self.outlier_threshold = FEATURE_CONFIG['outlier_threshold']
        self.ts_model = TimeSeriesModel(MODEL_PARAMS['xgb'])
        self.fe = OutlierDetectionTransformer(
        window_size=FEATURE_CONFIG['window_size'],
        outlier_threshold=FEATURE_CONFIG['outlier_threshold']
    )
    

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_features_before = self.fe.create_time_features(X['before'])
        X_features_before = self.ts_model.train_per_group(X_features_before)
        X_features_before = self.fe.detect_correct_outliers(X_features_before)
        X_features_after = self.fe.create_time_features( X['after'])
        X_features_after = self.ts_model.train_per_group(X_features_after)
        X_features_after = self.fe.detect_correct_outliers(X_features_after)
        return {'before':X_features_before, 'after':X_features_after}
