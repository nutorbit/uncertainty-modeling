import scipy
import numpy as np
import pandas as pd

from typing import List, Tuple, Any, Callable, Union
from collections import namedtuple

from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import PowerTransformer, OneHotEncoder

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


Encoder = namedtuple("Encoder", "cat num")


def safe_df(df: Any, cols: List[str] = None) -> pd.DataFrame:
    """
    Make sure that the dataframe will be in proper structure
    
    Args:
        df: Dataframe
        cols: list of the column name
        
    Returns:
        dataframe
    """
    
    if isinstance(df, scipy.sparse._csr.csr_matrix):
        df = df.toarray()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df, columns=cols)
    return df.reset_index(drop=True)


def encode_categorical_features(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, ...]:
    """
    Encode the categorical features using Target Encooder
    
    Args:
        df: Dataframe
        feature_cols: features
        target_col: target column name
        
    Returns:
        Tuple of dataframe and encoder
    """
    
    if not feature_cols:
        return None, None
    df_encoded = df.copy()
    encoder = TargetEncoder()
    df_encoded = encoder.fit_transform(df_encoded[feature_cols], df_encoded[target_col])
    return safe_df(df_encoded, feature_cols), encoder


def encode_categorical_features_v2(df: pd.DataFrame, feature_cols: List[str], target_col: str = None) -> Tuple[pd.DataFrame, ...]:
    """
    Encode the categorical features using OneHot Encooder
    
    Args:
        df: Dataframe
        feature_cols: features
        target_col: target column name (optional)
        
    Returns:
        Tuple of dataframe and encoder
    """
    
    if not feature_cols:
        return None, None
    df_encoded = df.copy()
    encoder = OneHotEncoder()
    df_encoded = encoder.fit_transform(df_encoded[feature_cols])
    return safe_df(df_encoded), encoder


def encode_numerical_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, ...]:
    """
    Encode the numerical features using BoxCox transform
    
    Args:
        df: Dataframe
        feature_cols: features
        
    Returns:
        Tuple of dataframe and encoder
    """
    
    if not feature_cols:
        return None, None
    df_encoded = df.copy()
    encoder = PowerTransformer()
    df_encoded = encoder.fit_transform(df_encoded[feature_cols])
    return safe_df(df_encoded, feature_cols), encoder


def encode_all(
    df: pd.DataFrame,
    cat_encoder_fn: Callable,
    num_encoder_fn: Callable,
    cat_features: List[str],
    num_features: List[str],
    target: str
) -> Tuple[pd.DataFrame, Encoder]:
    """
    Encode all features
    
    Args:
        df: Dataframe
        cat_encoder_fn: function to encode categorical features
        num_encoder_fn: function to encode numerical features
        cat_features: list of categorical features
        num_features: list of numerical features
        target: target column
        
    Returns:
        Tuple of dataframe and encoders
    """
    
    df_copy = df.reset_index(drop=True).copy()
    
    df_encoded_cat, cat_encoder = cat_encoder_fn(df_copy, cat_features, target)
    df_encoded_num, num_encoder = num_encoder_fn(df_copy, num_features)
    
    df_encoded = pd.concat([df_encoded_cat, df_encoded_num, df_copy[[target]]], axis=1)
    
    return df_encoded, Encoder(cat_encoder, num_encoder)


def apply_encoder(df: pd.DataFrame, enc: Union[TargetEncoder, PowerTransformer], feature_cols: List[str]) -> pd.DataFrame:
    """
    Apply encoder to dataframe
    
    Args:
        df: Dataframe
        enc: encoder instance
        feature_cols: features
        
    Returns:
        encoded dataframe
    """
    
    if not feature_cols:
        return None
    df_encoded = df.copy()
    df_encoded = enc.transform(df_encoded[feature_cols])
    return safe_df(df_encoded, cols=feature_cols if df_encoded.shape[1] == len(feature_cols) else None)


def apply_encoders(
    df: pd.DataFrame, 
    encoder: Encoder, 
    cat_features: List[str], 
    num_features: List[str], 
    target: str
) -> pd.DataFrame:
    """
    Apply all encoder to dataframe
    
    Args:
        df: Dataframe
        enc: encoder instance
        cat_features: categorical features
        num_features: numerical features
        target: target column
        
    Returns:
        encoded dataframe
    """
    
    
    df_copy = df.reset_index(drop=True).copy()
    
    df_encoded_cat = apply_encoder(df_copy, encoder.cat, cat_features)
    df_encoded_num = apply_encoder(df_copy, encoder.num, num_features)
    
    df_encoded = pd.concat([df_encoded_cat, df_encoded_num, df_copy[[target]]], axis=1)
    
    return df_encoded


def rollout_result(
    model_fn: Callable, 
    df: pd.DataFrame,
    cat_features: List[str],
    num_features: List[str],
    target: str,
    metric_fn: Callable = accuracy_score,
    use_onehot: bool = False,
    n_splits: int = 5,
    seed: int = 123
) -> Tuple[float, ...]:
    """
    Evaluate the input model using cross validation
    
    Args:
        model_fn: model function
        df: Dataframe
        cat_features: categorical features
        num_features: numerical features
        target: target column
        metric_fn: metric function that you want to measure on
        use_onehot: flag that indicate whether using one-hot encoder for cateogircal feature or not
        n_splits: number of k for cross validation
        seed: random state
        
    Returns:
        Tuple of mean of the score and best model
    """
    
    df_copy = df.copy()
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    models = []

    for train_idx, test_idx in kf.split(df_copy):
        train_df, test_df = df_copy.iloc[train_idx].reset_index(drop=True), df_copy.iloc[test_idx].reset_index(drop=True)

        # encode training set
        cat_encoder_fn = encode_categorical_features_v2 if use_onehot else encode_categorical_features
        num_encoder_fn = encode_numerical_features
        train_encoded, encoder = encode_all(train_df, cat_encoder_fn, num_encoder_fn, cat_features, num_features, target)

        # train the model
        model = model_fn()
        _ = model.fit(train_encoded.drop(target, axis=1), train_encoded[target])

        # encode testing set
        test_encoded = apply_encoders(test_df, encoder, cat_features, num_features, target)
        
        # evaluate
        pred = model.predict(test_encoded.drop(target, axis=1))
        y_true = test_encoded[[target]]
        
        scores.append(metric_fn(y_true, pred))
        models.append(model)

    best_idx = np.argmax(scores)
        
    return np.mean(scores), models[best_idx]
