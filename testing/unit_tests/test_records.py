# test_records.py
import pytest
import toml
import tensorflow as tf
import os
from steps.load_data import build_loader


@pytest.fixture(scope="module")
def params(config_path):
    """Loads parameters from the config file specified via command-line."""
    if not os.path.exists(config_path):
        pytest.skip(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        return toml.load(f)

@pytest.fixture(scope="module")
def data_loaders(params):
    """Uses the project's build_loader function to create data loaders."""
    data_path = params['target']['path']
    if not os.path.isdir(data_path):
        pytest.skip(f"Data directory '{data_path}' from config not found.")

    data_path = os.path.join(data_path, 'fold_0')

    try:
        params_data = {
            'probed':0.5,
            'rs': 0.20,
            'same': 0.20,
            'window_size':200,
            'arch':'base'
        }
        
        loaders = build_loader(
            data_path=data_path,
            params=params_data,
            batch_size=16,
            debug=False,
            return_test=False,
            repeat=1,
            sampling=True
        )

        return loaders
    except Exception as e:
        pytest.fail(f"build_loader failed with error: {e}")

# --- Test functions ---
def test_batch_shapes(data_loaders, params):
    test_loader = data_loaders['train']
    first_batch = next(iter(test_loader))
    inputs, _ = first_batch
    assert inputs['input'].shape[1] == 200 