import pandas as pd
import numpy as np

class OutlierDetectionTransformer:
    """Classe pour nettoyer les ts de bruits"""

    def __init__(self, window_size=3, outlier_threshold=3):
        self.window_size = window_size
        self.outlier_threshold = outlier_threshold
    
    def create_time_features(self, df:pd.DataFrame):
        """Crée les features temporelles de base"""
        df = df.copy()
        df['time_lag1'] = df.groupby('id')['value'].shift(1)
        df['rolling_mean'] = self._rolling_statistic(df, 'mean')
        df['rolling_std'] = self._rolling_statistic(df, 'std')
        return df.dropna()
    
    def _rolling_statistic(self, df, stat):
        """Calcule les statistiques glissantes"""
        return df.groupby('id')['value'].rolling(
            window=self.window_size
        ).agg(stat).reset_index(level=0, drop=True)
    
    def detect_correct_outliers(self, df):
        """Détecte et corrige les outliers"""
        df = df.copy()
        df['abs_error'] = np.abs(df['value'] - df['predicted'])
        df['is_outlier'] = df['abs_error'] > self.outlier_threshold * df['rolling_std']
        
        df['corrected_value'] = df['value']
        outlier_mask = df['is_outlier']
        df.loc[outlier_mask, 'corrected_value'] = df.loc[outlier_mask, 'rolling_mean']
        
        return df