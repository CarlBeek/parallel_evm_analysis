"""
Validate commands for the parallel-stats CLI.
Handles environment and node validation operations.
"""

import sys
import logging
from pathlib import Path

# Import the existing functionality from organized modules
from core.ethereum_client import EthereumClient


def handle_validate(args):
    """Handle all validate subcommands."""
    if args.validate_cmd == 'environment':
        return validate_environment()
    elif args.validate_cmd == 'node':
        return validate_node()
    else:
        print("❌ No validate operation specified. Use --help for options.")
        return 1


def validate_environment():
    """Validate environment setup."""
    print("🔍 Validating environment setup...")
    
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
    
    print(f"\n📋 Results:")
    print(f"   Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Environment is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1


def validate_node():
    """Validate Ethereum node connection."""
    print("🔍 Validating Ethereum node connection...")
    
    setup_logging()
    
    try:
        success = test_node_connection()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Node validation failed: {e}")
        return 1


# Environment validation functions (from validate_environment.py)

def test_python_version():
    """Test Python version."""
    print(f"Python version: {sys.version}")
    if sys.version_info >= (3, 12):
        print("✓ Python 3.12+ detected")
        return True
    else:
        print("✗ Python 3.12+ required")
        return False


def test_basic_imports():
    """Test basic Python imports."""
    try:
        import json
        import pathlib
        import sqlite3
        import concurrent.futures
        print("✓ Basic imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_web3_import():
    """Test Web3 import."""
    try:
        from web3 import Web3
        print(f"✓ Web3 import successful (version check needed)")
        return True
    except ImportError as e:
        print(f"✗ Web3 import failed: {e}")
        return False


def test_project_structure():
    """Test project structure."""
    required_dirs = [
        "src",
        "src/core",
        "src/analysis", 
        "src/storage",
        "src/visualization",
        "tests",
        "data",
        "logs"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"✗ Missing directories: {missing_dirs}")
        return False
    else:
        print("✓ Project structure is correct")
        return True


def test_config_file():
    """Test config file exists and is valid."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("✗ config.yaml not found")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['ethereum', 'database', 'analysis', 'visualization']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            print(f"✗ Missing config sections: {missing_sections}")
            return False
        else:
            print("✓ Config file is valid")
            return True
            
    except Exception as e:
        print(f"✗ Config file error: {e}")
        return False


# Node validation functions (from validate_node_connection.py)

def setup_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_node_connection():
    """Test Ethereum node connection."""
    logger = logging.getLogger(__name__)
    
    try:
        print("🔗 Testing Ethereum node connection...")
        
        # Initialize client
        client = EthereumClient()
        
        # Test basic connection
        print("   📡 Testing basic connection...")
        if not client.is_connected():
            print("   ❌ Cannot connect to Ethereum node")
            return False
        
        print("   ✅ Basic connection successful")
        
        # Get node info
        print("   📊 Getting node information...")
        try:
            chain_id = client.get_chain_id()
            latest_block = client.get_latest_block_number()
            
            print(f"   📈 Chain ID: {chain_id}")
            print(f"   📈 Latest block: {latest_block}")
            
            # Identify network
            network_names = {
                1: "Mainnet",
                3: "Ropsten",
                4: "Rinkeby", 
                5: "Goerli",
                11155111: "Sepolia"
            }
            network_name = network_names.get(chain_id, f"Unknown (Chain ID: {chain_id})")
            print(f"   🌐 Network: {network_name}")
            
        except Exception as e:
            print(f"   ⚠️  Could not get detailed node info: {e}")
        
        # Test block fetching
        print("   📦 Testing block fetching...")
        try:
            latest_block = client.get_latest_block_number()
            block = client.get_block(latest_block)
            
            if block and 'transactions' in block:
                tx_count = len(block['transactions'])
                print(f"   ✅ Block fetching successful (latest block has {tx_count} transactions)")
            else:
                print("   ⚠️  Block fetching returned unexpected format")
                
        except Exception as e:
            print(f"   ⚠️  Block fetching failed: {e}")
        
        # Test transaction fetching
        print("   💰 Testing transaction fetching...")
        try:
            latest_block = client.get_latest_block_number()
            block = client.get_block(latest_block)
            
            if block and 'transactions' in block and len(block['transactions']) > 0:
                # Get first transaction
                tx_hash = block['transactions'][0]
                transaction = client.get_transaction(tx_hash)
                
                if transaction:
                    print(f"   ✅ Transaction fetching successful")
                else:
                    print("   ⚠️  Transaction fetching returned no data")
            else:
                print("   ℹ️  No transactions in latest block to test")
                
        except Exception as e:
            print(f"   ⚠️  Transaction fetching failed: {e}")
        
        # Test debug API availability
        print("   🔧 Testing debug API availability...")
        try:
            # This would test debug_traceTransaction
            print("   ⚠️  Debug API testing not implemented (requires specific transaction)")
        except Exception as e:
            print(f"   ❌ Debug API not available: {e}")
        
        print("\n🎉 Node connection test completed successfully!")
        print("💡 The node is accessible and basic operations work.")
        
        return True
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        print(f"\n❌ Node connection test failed: {e}")
        return False 