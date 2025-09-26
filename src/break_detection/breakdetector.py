from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from config import MODEL_PARAMS

class BreakDetector:
    """Classe pour détecter les breaks statistiques"""
    
    def __init__(self, model_type='mlp', optimize_features=True, max_features=None):
        self.model_type = model_type
        self.optimize_features = optimize_features
        self.max_features = max_features
        self.scaler = StandardScaler()
        self.feature_importances_ = None
        self.optimal_features_ = None
        self.selected_feature_indices_ = None
        self.model_ = None
        self.is_trained_ = False
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entraîne le modèle avec optimisation des features si demandé
        """
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, 
                test_size=0.3, 
                random_state=1,
            )
        
        X_tr_scaled = self.scaler.fit_transform(X_train)
        X_va_scaled = self.scaler.transform(X_val)
        
        if self.optimize_features:
            optimal_n, _, sorted_idx = self.optimize_features(
                X_tr_scaled, X_va_scaled, y_train, y_val, self.max_features
            )
            self.selected_feature_indices_ = sorted_idx[:optimal_n]
            self.optimal_features_ = optimal_n
        else:
            self.selected_feature_indices_ = np.arange(X_tr_scaled.shape[1])
            self.optimal_features_ = X_tr_scaled.shape[1]
        
        X_tr_final = X_tr_scaled[:, self.selected_feature_indices_]
        X_va_final = X_va_scaled[:, self.selected_feature_indices_]
        
        self.model_ = self._create_model()
        self.model_.fit(X_tr_final, y_train)
        
        val_score = self._validate_model(X_va_final, y_val)
        self.is_trained_ = True
        
        return {
            'optimal_features': self.optimal_features_,
            'validation_score': val_score,
            'feature_importances': self.feature_importances_
        }
    
    def predict(self, X):
        """
        Prédit les breaks sur de nouvelles données
        """
        if not self.is_trained_:
            raise ValueError("Le modèle n'est pas entraîné. Appelez train() d'abord.")
        
        X_scaled = self.scaler.transform(X)
        X_final = X_scaled[:, self.selected_feature_indices_]
        
        if hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(X_final)[:, 1]
        else:
            return self.model_.predict(X_final)
    
    def optimize_features(self, X_train, X_test, y_train, y_test, max_features=None):
        """Trouve le nombre optimal de features"""
        if max_features is None:
            max_features = X_train.shape[1]
            
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        self.feature_importances_ = rf.feature_importances_
        sorted_idx = np.argsort(self.feature_importances_)[::-1]
        
        scores = []
        for f in range(1, min(max_features + 1, X_train.shape[1] + 1)):
            X_train_sub = X_train[:, sorted_idx[:f]]
            X_test_sub = X_test[:, sorted_idx[:f]]
            
            score = self._evaluate_model(X_train_sub, X_test_sub, y_train, y_test)
            scores.append(score)
        
        optimal_n = np.argmax(scores) + 1
        return optimal_n, scores, sorted_idx
    
    def _evaluate_model(self, X_train, X_test, y_train, y_test):
        """Évalue un modèle avec les features données"""
        mlp = MLPClassifier(**MODEL_PARAMS['mlp'])
        mlp.fit(X_train, y_train)
        y_proba = mlp.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_proba)
    
    def _create_model(self):
        """Crée le modèle selon le type spécifié"""
        if self.model_type == 'mlp':
            return MLPClassifier(**MODEL_PARAMS['mlp'])
        elif self.model_type == 'rf':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Type de modèle non supporté")
    
    def _validate_model(self, X_val, y_val):
        """Valide le modèle entraîné"""
        if hasattr(self.model_, 'predict_proba'):
            y_proba = self.model_.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_proba)
        else:
            y_pred = self.model_.predict(X_val)
            return roc_auc_score(y_val, y_pred)