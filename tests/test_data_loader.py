import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from typing import List, Optional, Union

# Assume data_loader.py contains these functions:
# We define them here for the test file to be self-contained and runnable.

# --- Mock data_loader Module ---

class DataLoaderError(Exception):
    """Custom exception for data loading errors."""
    pass

def load_csv(filepath: str) -> pd.DataFrame:
    """Loads data from a CSV file into a pandas DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            print(f"Warning: Loaded empty DataFrame from {filepath}")
        return df
    except pd.errors.EmptyDataError:
        print(f"Warning: CSV file is empty: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        raise DataLoaderError(f"Error loading CSV file {filepath}: {e}") from e

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mean',
    subset: Optional[List[str]] = None
) -> pd.DataFrame:
    """Handles missing values in a DataFrame."""
    df_copy = df.copy()
    columns_to_process = subset if subset else df_copy.select_dtypes(include=np.number).columns

    if strategy == 'mean':
        for col in columns_to_process:
            if df_copy[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    mean_val = df_copy[col].mean()
                    df_copy[col].fillna(mean_val, inplace=True)
    elif strategy == 'median':
        for col in columns_to_process:
            if df_copy[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    median_val = df_copy[col].median()
                    df_copy[col].fillna(median_val, inplace=True)
    elif strategy == 'mode':
         columns_to_process = subset if subset else df_copy.columns # Mode can apply to non-numeric
         for col in columns_to_process:
             if df_copy[col].isnull().any():
                 mode_val = df_copy[col].mode()[0] # mode() returns Series
                 df_copy[col].fillna(mode_val, inplace=True)
    elif strategy == 'drop':
        df_copy.dropna(subset=subset, inplace=True)
    elif strategy == 'constant':
        raise ValueError("Strategy 'constant' requires a 'fill_value' argument (not implemented in this mock).")
    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")

    return df_copy

def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'onehot'
) -> pd.DataFrame:
    """Encodes categorical features."""
    df_copy = df.copy()
    if method == 'onehot':
        df_copy = pd.get_dummies(df_copy, columns=columns, drop_first=True)
    elif method == 'label':
        # Basic label encoding for simplicity in mock
        for col in columns:
            df_copy[col] = df_copy[col].astype('category').cat.codes
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    return df_copy

def scale_numerical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'minmax'
) -> pd.DataFrame:
    """Scales numerical features."""
    df_copy = df.copy()
    if method == 'minmax':
        for col in columns:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                min_val = df_copy[col].min()
                max_val = df_copy[col].max()
                range_val = max_val - min_val
                if range_val == 0:
                     df_copy[col] = 0.0 # Avoid division by zero if all values are the same
                else:
                    df_copy[col] = (df_copy[col] - min_val) / range_val
            else:
                 print(f"Warning: Column {col} is not numeric and cannot be scaled.")

    elif method == 'standard':
        for col in columns:
             if pd.api.types.is_numeric_dtype(df_copy[col]):
                mean_val = df_copy[col].mean()
                std_val = df_copy[col].std()
                if std_val == 0:
                    df_copy[col] = 0.0 # Avoid division by zero
                else:
                    df_copy[col] = (df_copy[col] - mean_val) / std_val
             else:
                 print(f"Warning: Column {col} is not numeric and cannot be scaled.")
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    return df_copy


# --- Test Fixtures ---

@pytest.fixture(scope="function")
def sample_data_dict():
    """Provides a dictionary with sample data including NaNs."""
    return {
        'numeric_col1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'numeric_col2': [10.0, 20.0, 30.0, 40.0, 50.0],
        'categorical_col1': ['A', 'B', 'A', 'C', 'B'],
        'categorical_col2': ['X', 'Y', np.nan, 'X', 'Y'],
        'constant_col': [5, 5, 5, 5, 5]
    }

@pytest.fixture(scope="function")
def sample_dataframe(sample_data_dict):
    """Provides a pandas DataFrame from sample_data_dict."""
    return pd.DataFrame(sample_data_dict)

@pytest.fixture(scope="function")
def temp_csv_file(sample_dataframe):
    """Creates a temporary CSV file with sample data."""
    # Using NamedTemporaryFile for automatic cleanup
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".csv", newline='') as temp_file:
        sample_dataframe.to_csv(temp_file.name, index=False)
        file_path = temp_file.name
    # Yield the path, then clean up after the test
    yield file_path
    os.remove(file_path)

@pytest.fixture(scope="function")
def empty_temp_csv_file():
    """Creates an empty temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".csv") as temp_file:
        file_path = temp_file.name
    yield file_path
    os.remove(file_path)


# --- Test Cases ---

# 1. Data Loading Tests
def test_load_csv_success(temp_csv_file, sample_dataframe):
    """Test loading a valid CSV file."""
    loaded_df = load_csv(temp_csv_file)
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)

def test_load_csv_file_not_found():
    """Test loading a non-existent CSV file raises FileNotFoundError."""
    non_existent_file = "non_existent_file_12345.csv"
    with pytest.raises(FileNotFoundError):
        load_csv(non_existent_file)

def test_load_empty_csv(empty_temp_csv_file):
    """Test loading an empty CSV file returns an empty DataFrame."""
    loaded_df = load_csv(empty_temp_csv_file)
    assert loaded_df.empty
    assert isinstance(loaded_df, pd.DataFrame)

# 2. Data Preprocessing Tests
def test_handle_missing_values_mean(sample_dataframe):
    """Test filling missing numeric values with the mean."""
    processed_df = handle_missing_values(sample_dataframe, strategy='mean', subset=['numeric_col1'])
    expected_mean = sample_dataframe['numeric_col1'].mean()
    assert not processed_df['numeric_col1'].isnull().any()
    assert processed_df['numeric_col1'].iloc[2] == pytest.approx(expected_mean)
    # Check other columns remain unchanged or handled appropriately
    assert processed_df['numeric_col2'].equals(sample_dataframe['numeric_col2'])
    assert processed_df['categorical_col2'].isnull().iloc[2] # NaN should persist if not in subset

def test_handle_missing_values_median(sample_dataframe):
    """Test filling missing numeric values with the median."""
    processed_df = handle_missing_values(sample_dataframe, strategy='median', subset=['numeric_col1'])
    expected_median = sample_dataframe['numeric_col1'].median()
    assert not processed_df['numeric_col1'].isnull().any()
    assert processed_df['numeric_col1'].iloc[2] == pytest.approx(expected_median)

def test_handle_missing_values_mode(sample_dataframe):
    """Test filling missing categorical values with the mode."""
    processed_df = handle_missing_values(sample_dataframe, strategy='mode', subset=['categorical_col2'])
    expected_mode = sample_dataframe['categorical_col2'].mode()[0] # Should be 'X' or 'Y'
    assert not processed_df['categorical_col2'].isnull().any()
    assert processed_df['categorical_col2'].iloc[2] == expected_mode
    # Check other columns remain unchanged or handled appropriately
    assert processed_df['numeric_col1'].isnull().iloc[2] # NaN should persist if not in subset

def test_handle_missing_values_drop(sample_dataframe):
    """Test dropping rows with missing values."""
    # Drop rows where 'numeric_col1' is NaN
    processed_df_numeric = handle_missing_values(sample_dataframe, strategy='drop', subset=['numeric_col1'])
    assert processed_df_numeric.shape[0] == 4
    assert 2 not in processed_df_numeric.index # Row index 2 should be dropped

    # Drop rows where 'categorical_col2' is NaN
    processed_df_categorical = handle_missing_values(sample_dataframe, strategy='drop', subset=['categorical_col2'])
    assert processed_df_categorical.shape[0] == 4
    assert 2 not in processed_df_categorical.index # Row index 2 should be dropped

    # Drop rows where any value is NaN (default subset=None behavior for dropna)
    processed_df_any = handle_missing_values(sample_dataframe.copy(), strategy='drop') # Need copy as dropna works inplace in mock
    assert processed_df_any.shape[0] == 3 # Rows with NaN in numeric_col1 or categorical_col2 are dropped
    assert 2 not in processed_df_any.index

def test_handle_missing_values_invalid_strategy(sample_dataframe):
    """Test using an invalid strategy raises ValueError."""
    with pytest.raises(ValueError):
        handle_missing_values(sample_dataframe, strategy='invalid_strategy')

def test_handle_missing_values_no_nans(sample_dataframe):
    """Test handling missing values on a DataFrame with no NaNs."""
    df_no_nans = sample_dataframe.dropna().reset_index(drop=True)
    processed_df = handle_missing_values(df_no_nans.copy(), strategy='mean')
    pd.testing.assert_frame_equal(processed_df, df_no_nans)


# 3. Data Transformation Tests
def test_encode_categorical_onehot(sample_dataframe):
    """Test one-hot encoding for categorical columns."""
    df_no_nans = sample_dataframe.dropna(subset=['categorical_col1', 'categorical_col2']).reset_index(drop=True)
    columns_to_encode = ['categorical_col1', 'categorical_col2']
    encoded_df = encode_categorical(df_no_nans, columns=columns_to_encode, method='onehot')

    # Check original columns are removed
    assert 'categorical_col1' not in encoded_df.columns
    assert 'categorical_col2' not in encoded_df.columns

    # Check new columns are added (drop_first=True means k-1 columns per original)
    # unique values: cat1={'A', 'B', 'C'}, cat2={'X', 'Y'}
    assert 'categorical_col1_B' in encoded_df.columns
    assert 'categorical_col1_C' in encoded_df.columns
    assert 'categorical_col1_A' not in encoded_df.columns # Dropped first ('A')
    assert 'categorical_col2_Y' in encoded_df.columns
    assert 'categorical_col2_X' not in encoded_df.columns # Dropped first ('X')

    # Check shape
    assert encoded_df.shape[0] == df_no_nans.shape[0]
    # Original cols - encoded cols + new one-hot cols
    # 5 - 2 + (3-1) + (2-1) = 5 - 2 + 2 + 1 = 6 columns
    assert encoded_df.shape[1] == 6

    # Check values (example)
    # Original row 0: cat1='A', cat2='X' -> cat1_B=0, cat1_C=0, cat2_Y=0
    assert encoded_df.loc[0, 'categorical_col1_B'] == 0
    assert encoded_df.loc[0, 'categorical_col1_C'] == 0
    assert encoded_df.loc[0, 'categorical_col2_Y'] == 0
    # Original row 1: cat1='B', cat2='Y' -> cat1_B=1, cat1_C=0, cat2_Y=1
    assert encoded_df.loc[1, 'categorical_col1_B'] == 1
    assert encoded_df.loc[1, 'categorical_col1_C'] == 0
    assert encoded_df.loc[1, 'categorical_col2_Y'] == 1


def test_encode_categorical_label(sample_dataframe):
    """Test label encoding for categorical columns."""
    df_no_nans = sample_dataframe.dropna(subset=['categorical_col1', 'categorical_col2']).reset_index(drop=True)
    columns_to_encode = ['categorical_col1', 'categorical_col2']
    encoded_df = encode_categorical(df_no_nans.copy(), columns=columns_to_encode, method='label')

    # Check columns still exist but are numeric
    assert 'categorical_col1' in encoded_df.columns
    assert 'categorical_col2' in encoded_df.columns
    assert pd.api.types.is_numeric_dtype(encoded_df['categorical_col1'])
    assert pd.api.types.is_numeric_dtype(encoded_df['categorical_col2'])

    # Check encoding mapping (depends on order, usually alphabetical)
    # cat1: A=0, B=1, C=2
    # cat2: X=0, Y=1
    expected_cat1 = pd.Series([0, 1, 0, 1], name='categorical_col1') # A, B, A, B (after dropna)
    expected_cat2 = pd.Series([0, 1, 0, 1], name='categorical_col2') # X, Y, X, Y (after dropna)

    pd.testing.assert_series_equal(encoded_df['categorical_col1'], expected_cat1, check_dtype=False, check_names=False)
    pd.testing.assert_series_equal(encoded_df['categorical_col2'], expected_cat2, check_dtype=False, check_names=False)


def test_scale_numerical_minmax(sample_dataframe):
    """Test Min-Max scaling for numerical columns."""
    df_no_nans = sample_dataframe.dropna().reset_index(drop=True)
    columns_to_scale = ['numeric_col1', 'numeric_col2']
    scaled_df = scale_numerical(df_no_nans.copy(), columns=columns_to_scale, method='minmax')

    # Check values are between 0 and 1
    for col in columns_to_scale:
        assert scaled_df[col].min() >= 0.0
        assert scaled_df[col].max() <= 1.0
        # Check specific values
        # numeric_col1 original (no NaNs): [1.0, 2.0, 4.0, 5.0] -> min=1, max=5, range=4
        # Scaled: (1-1)/4=0, (2-1)/4=0.25, (4-1)/4=0.75, (5-1)/4=1.0
        expected_col1 = pd.Series([0.0, 0.25, 0.75, 1.0], name='numeric_col1')
        pd.testing.assert_series_equal(scaled_df['numeric_col1'], expected_col1, check_names=False)

        # numeric_col2 original (no NaNs): [10.0, 20.0, 40.0, 50.0] -> min=10, max=50, range=40
        # Scaled: (10-10)/40=0, (20-10)/40=0.25, (40-10)/40=0.75, (50-10)/40=1.0
        expected_col2 = pd.Series([0.0, 0.25, 0.75, 1.0], name='numeric_col2')
        pd.testing.assert_series_equal(scaled_df['numeric_col2'], expected_col2, check_names=False)

def test_scale_numerical_standard(sample_dataframe):
    """Test Standard scaling for numerical columns."""
    df_no_nans = sample_dataframe.dropna().reset_index(drop=True)
    columns_to_scale = ['numeric_col1', 'numeric_col2']
    scaled_df = scale_numerical(df_no_nans.copy(), columns=columns_to_scale, method='standard')

    # Check mean is close to 0 and std dev is close to 1
    for col in columns_to_scale:
        assert scaled_df[col].mean() == pytest.approx(0.0, abs=1e-9)
        assert scaled_df[col].std() == pytest.approx(1.0, abs=1e-9)

def test_scale_numerical_constant_column(sample_dataframe):
    """Test scaling a column with constant values."""
    df = sample_dataframe.copy()
    columns_to_scale = ['constant_col']

    # MinMax scaling
    scaled_df_minmax = scale_numerical(df.copy(), columns=columns_to_scale, method='minmax')
    assert scaled_df_minmax['constant_col'].nunique() == 1
    assert scaled_df_minmax['constant_col'].iloc[0] == 0.0 # Should scale to 0

    # Standard scaling
    scaled_df_standard = scale_numerical(df.copy(), columns=columns_to_scale, method='standard')
    assert scaled_df_standard['constant_col'].nunique() == 1
    assert scaled_df_standard['constant_col'].iloc[0] == 0.0 # Should scale to 0

def test_scale_non_numeric_column(sample_dataframe, capsys):
    """Test that attempting to scale a non-numeric column prints a warning."""
    df = sample_dataframe.copy()
    columns_to_scale = ['categorical_col1']
    scaled_df = scale_numerical(df.copy(), columns=columns_to_scale, method='minmax')

    # Check that the original non-numeric column is unchanged
    pd.testing.assert_series_equal(df['categorical_col1'], scaled_df['categorical_col1'])

    # Check that a warning was printed (optional, depends on implementation)
    captured = capsys.readouterr()
    assert "Warning: Column categorical_col1 is not numeric" in captured.out

def test_invalid_scaling_method(sample_dataframe):
    """Test using an invalid scaling method raises ValueError."""
    with pytest.raises(ValueError):
        scale_numerical(sample_dataframe, columns=['numeric_col1'], method='invalid_scaler')

def test_invalid_encoding_method(sample_dataframe):
    """Test using an invalid encoding method raises ValueError."""
    with pytest.raises(ValueError):
        encode_categorical(sample_dataframe, columns=['categorical_col1'], method='invalid_encoder')