import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--config-path", 
        action="store", 
        default="./data/config.toml", 
        help="Path to the main config.toml file"
    )

@pytest.fixture(scope="module")  # Add scope="module"
def config_path(request):
    """Creates a fixture to get the value of --config-path."""
    return request.config.getoption("--config-path")