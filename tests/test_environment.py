"""
Test environment setup and basic functionality.
"""

import sys
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_python_version():
    """Test that we're using Python 3.12+"""
    assert sys.version_info >= (3, 12), f"Python 3.12+ required, got {sys.version_info}"


def test_imports():
    """Test that key dependencies can be imported"""
    try:
        import web3
        import pandas
        import numpy
        import networkx
        import matplotlib
        import plotly
        import sqlalchemy
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import required dependency: {e}")


def test_project_structure():
    """Test that project directories exist"""
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        "src",
        "src/core",
        "src/analysis", 
        "src/visualization",
        "src/storage",
        "tests",
        "data",
        "logs"
    ]
    
    for dir_path in required_dirs:
        assert (project_root / dir_path).exists(), f"Directory {dir_path} does not exist"


def test_config_file():
    """Test that configuration file exists"""
    project_root = Path(__file__).parent.parent
    config_file = project_root / "config.yaml"
    assert config_file.exists(), "config.yaml file does not exist"


if __name__ == "__main__":
    pytest.main([__file__]) 