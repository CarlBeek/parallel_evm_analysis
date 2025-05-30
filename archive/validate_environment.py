#!/usr/bin/env python3
"""
Basic environment test without pytest dependencies.
"""

import sys
from pathlib import Path

def test_python_version():
    """Test that we're using Python 3.12+"""
    print(f"Python version: {sys.version}")
    assert sys.version_info >= (3, 12), f"Python 3.12+ required, got {sys.version_info}"
    print("✓ Python version check passed")
    return True

def test_basic_imports():
    """Test that basic dependencies can be imported"""
    try:
        import pandas
        print(f"✓ pandas {pandas.__version__}")
        
        import numpy
        print(f"✓ numpy {numpy.__version__}")
        
        import networkx
        print(f"✓ networkx {networkx.__version__}")
        
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
        
        import sqlalchemy
        print(f"✓ sqlalchemy {sqlalchemy.__version__}")
        
        print("✓ Basic imports successful")
        
    except ImportError as e:
        print(f"✗ Failed to import dependency: {e}")
        return False
    return True

def test_web3_import():
    """Test web3 import separately due to potential issues"""
    try:
        import web3
        print(f"✓ web3 {web3.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import web3: {e}")
        return False

def test_project_structure():
    """Test that project directories exist"""
    project_root = Path(__file__).parent
    
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
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} does not exist")
            all_exist = False
    
    return all_exist

def test_config_file():
    """Test that configuration file exists"""
    project_root = Path(__file__).parent
    config_file = project_root / "config.yaml"
    if config_file.exists():
        print("✓ config.yaml exists")
        return True
    else:
        print("✗ config.yaml does not exist")
        return False

def main():
    """Run all tests"""
    print("=== Environment Setup Test ===\n")
    
    tests = [
        ("Python Version", test_python_version),
        ("Basic Imports", test_basic_imports),
        ("Web3 Import", test_web3_import),
        ("Project Structure", test_project_structure),
        ("Config File", test_config_file),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} failed")
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
    
    print(f"\n=== Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Environment is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 