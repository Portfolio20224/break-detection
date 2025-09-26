from sklearn.base import BaseEstimator, TransformerMixin
from break_detection import TimeSeriesFeatureTransformer
class BreakFeatureExtractorStep(BaseEstimator, TransformerMixin):
    """Extracteur de features pour les breaks"""
    
    def __init__(self):
        self.advanced_fe = TimeSeriesFeatureTransformer()    
    def fit(self, X, y=None):

        return self
    
    def transform(self, X):
        if isinstance(X, dict):
            return self.advanced_fe.extract_break_features(X['before'], X['after'])
        else:
            raise ValueError("X doit être un dictionnaire avec les clés 'before' et 'after'")
