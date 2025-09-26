from sklearn.base import BaseEstimator
from break_detection import BreakDetector


class BreakDetectorEstimator(BaseEstimator):
    """Estimateur final pour la détection de breaks"""
    
    def __init__(self):
        self.detector = BreakDetector()
        self.optimal_n = None
        self.scores = None
        self.sorted_idx = None
    
    def fit(self, X, y):
        self.detector.train(X, y)
        
        self.optimal_n = self.detector.optimal_n
        self.scores = self.detector.scores
        self.sorted_idx = self.detector.sorted_idx
        
        return self
    
    def predict(self, X):
        return self.detector.predict(X)
    
    def score(self, X, y=None):
        """Optionnel : on peut définir un score custom"""
        if self.scores is not None:
            return max(self.scores)
        else:
            raise ValueError("Le modèle doit être fit avant d'appeler score()")
