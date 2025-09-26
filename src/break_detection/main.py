from sklearn.pipeline import Pipeline
from .pipeline import FeatureEngineeringStep, BreakFeatureExtractorStep, BreakDetectorEstimator
from .config import MODEL_PARAMS, FEATURE_CONFIG
import pandas as pd

X_train = pd.read_parquet("chemin-du-fichier")
y_train = pd.read_parquet("chemin-du-fichier")
X_before = X_train[X_train['period'] == 0].copy()
X_after = X_train[X_train['period'] == 1].copy()

X_test = pd.read_parquet("chemin-du fichier")
y_test = pd.read_parquet("chemin-du-fichier")
X_test_before = X_test[X_test['period'] == 0].copy()
X_test_after = X_test[X_test['period'] == 1].copy()


def main():
    """Main"""
    
    pipeline = Pipeline([
        ('preprocessing', FeatureEngineeringStep(FEATURE_CONFIG, MODEL_PARAMS)),
        ('break_features', BreakFeatureExtractorStep()),
        ('detector', BreakDetectorEstimator())
    ])
    
    X_train = {
        'before': X_before,
        'after': X_after
    }
    
    X_test = {
        'before': X_test_before, 
        'after': X_test_after
    }
    
    pipeline.fit(X_train, y_train)
    
    predictions = pipeline.predict(X_test)
    
    score = pipeline.score(X_test, y_test)
    print(f"Score du pipeline: {score}")

if __name__ == "__main__":
    main()
