import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ranksums, ks_2samp, theilslopes, entropy
from tqdm import tqdm

class TimeSeriesFeatureTransformer:
    """Classe pour les features avancées de détection de breaks"""
    
    @staticmethod
    def spectral_entropy(series):
        psd = np.abs(np.fft.fft(series))**2
        psd_norm = psd / psd.sum() if psd.sum() != 0 else np.ones_like(psd) / len(psd)
        return entropy(psd_norm)
        
    @staticmethod
    def variance_ratio_test(series, window=10):
        """Test de ratio de variance pour détecter les breaks de volatilité"""
        if len(series) < 2 * window:
            return np.nan
        
        mid = len(series) // 2
        var1 = series.iloc[:mid].var()
        var2 = series.iloc[mid:].var()
        
        if var1 == 0 or var2 == 0:
            return np.nan
        
        return max(var1, var2) / min(var1, var2)
    
    @staticmethod
    def cusum_statistic(series):
        mean_val = series.mean()
        cusum = np.cumsum(series - mean_val)
        return np.max(np.abs(cusum)) / (series.std() * np.sqrt(len(series)))
    
    def extract_break_features(self, df_before, df_after):
        """Extrait les features de comparaison entre périodes"""
        features = []
        
        for id_ in tqdm(df_before.index.get_level_values('id').unique(), desc="Extracting features"):
            series_before = df_before.loc[id_]['corrected_value'].dropna()
            series_after = df_after.loc[id_]['corrected_value'].dropna()
            
            if len(series_before) == 0 or len(series_after) == 0:
                continue
                
            feature_dict = self._calculate_features(series_before, series_after)
            feature_dict['id'] = id_
            features.append(feature_dict)
        
        return pd.DataFrame(features).set_index('id')
    
    def _calculate_features(self, series_before, series_after):
        """Calcule toutes les features statistiques"""
        window = min(10, len(series_before)//2, len(series_after)//2)
        min_periods = max(5, window//10)
        
        slope_before = np.nan
        slope_after = np.nan
        entropy_ratio = np.nan
        
        if window > 1:
            slope_before = theilslopes(series_before.iloc[-window:])[0]
            slope_after = theilslopes(series_after.iloc[:window])[0]
            entropy_after = series_after.rolling(window=window, min_periods=min_periods).apply(self.spectral_entropy, raw=True).mean()
            entropy_before = series_before.rolling(window=window, min_periods=min_periods).apply(self.spectral_entropy, raw=True).mean()
            if entropy_before != 0:
                entropy_ratio = entropy_after / entropy_before
        
        combined_series = pd.concat([series_before, series_after])
        
        cusum_stat = self.cusum_statistic(combined_series)
        
        variance_ratio = self.variance_ratio_test(combined_series)
        
        acf_before = series_before.autocorr(lag=1) if len(series_before) > 1 else np.nan
        acf_after = series_after.autocorr(lag=1) if len(series_after) > 1 else np.nan
        
        q25_diff = series_after.quantile(0.25) - series_before.quantile(0.25)
        q75_diff = series_after.quantile(0.75) - series_before.quantile(0.75)
        
        t_stat = np.nan
        if len(series_before) > 1 and len(series_after) > 1:
            t_stat = ttest_ind(series_before, series_after, equal_var=False).statistic
        
        cv_before = series_before.std() / abs(series_before.mean()) if series_before.mean() != 0 else np.nan
        cv_after = series_after.std() / abs(series_after.mean()) if series_after.mean() != 0 else np.nan
        cv_change = cv_after - cv_before if not np.isnan(cv_before) and not np.isnan(cv_after) else np.nan
        
        return {
            'mean_diff': series_after.mean() - series_before.mean(),
            'volatility_change': series_after.std()/series_before.std() if series_before.std() != 0 else np.nan,
            'max_diff': series_after.max() - series_before.max(),
            'min_diff': series_after.min() - series_before.min(),
            'skew_diff': series_after.skew() - series_before.skew(),
            'kurt_diff': series_after.kurtosis() - series_before.kurtosis(),
            'median_diff': series_after.median() - series_before.median(),
            'acf_diff': acf_after - acf_before if not np.isnan(acf_before) and not np.isnan(acf_after) else np.nan,
            'ks_stat': ks_2samp(series_before, series_after).statistic,
            'mannwhitney_u': ranksums(series_before, series_after).statistic,
            'mean_before': series_before.mean(),
            'entropy_before': entropy(np.histogram(series_before, bins=20)[0]),
            'slope_change_robust': slope_after - slope_before if not np.isnan(slope_before) and not np.isnan(slope_after) else np.nan,
            'entropy_ratio': entropy_ratio,
            
            'cusum_statistic': cusum_stat,
            'variance_ratio': variance_ratio,
            'q25_diff': q25_diff,
            'q75_diff': q75_diff,
            't_statistic': t_stat,
            'cv_change': cv_change,
            
            'relative_mean_change': (series_after.mean() - series_before.mean()) / abs(series_before.mean()) if series_before.mean() != 0 else np.nan,
            'relative_std_change': (series_after.std() - series_before.std()) / series_before.std() if series_before.std() != 0 else np.nan,
            
            'p_value_ks': ks_2samp(series_before, series_after).pvalue,
            'p_value_mannwhitney': ranksums(series_before, series_after).pvalue if len(series_before) > 0 and len(series_after) > 0 else np.nan,
        }