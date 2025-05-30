#!/usr/bin/env python3
"""
Test script to verify connection to the Ethereum node and basic functionality.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.ethereum_client import EthereumClient

def setup_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_node_connection():
    """Test basic connection to the Ethereum node."""
    print("=== Ethereum Node Connection Test ===\n")
    
    try:
        # Initialize client
        print("1. Initializing Ethereum client...")
        client = EthereumClient()
        
        # Check connection
        print("2. Checking connection...")
        if client.is_connected():
            print("✓ Successfully connected to Ethereum node")
        else:
            print("✗ Failed to connect to Ethereum node")
            return False
        
        # Get node information
        print("\n3. Retrieving node information...")
        node_info = client.get_node_info()
        
        print(f"   Client Version: {node_info.get('client_version', 'Unknown')}")
        print(f"   Chain ID: {node_info.get('chain_id', 'Unknown')}")
        print(f"   Latest Block: {node_info.get('latest_block', 'Unknown')}")
        print(f"   Syncing: {node_info.get('syncing', 'Unknown')}")
        print(f"   Gas Price: {node_info.get('gas_price', 'Unknown')} wei")
        print(f"   Debug API Available: {node_info.get('debug_api_available', 'Unknown')}")
        
        # Test basic block fetching
        print("\n4. Testing block fetching...")
        latest_block_num = client.get_latest_block_number()
        print(f"   Latest block number: {latest_block_num}")
        
        # Fetch a recent block (without full transactions for speed)
        test_block_num = max(1, latest_block_num - 1)  # Get previous block to ensure it exists
        print(f"   Fetching block {test_block_num}...")
        
        block = client.get_block(test_block_num, full_transactions=False)
        print(f"   ✓ Block {test_block_num} retrieved successfully")
        print(f"   Block hash: {block.get('hash', 'Unknown')}")
        print(f"   Number of transactions: {len(block.get('transactions', []))}")
        print(f"   Gas used: {block.get('gasUsed', 'Unknown')}")
        print(f"   Gas limit: {block.get('gasLimit', 'Unknown')}")
        
        # Test transaction fetching if there are transactions
        transactions = block.get('transactions', [])
        if transactions:
            print("\n5. Testing transaction fetching...")
            tx_hash = transactions[0]  # Get first transaction hash
            print(f"   Fetching transaction {tx_hash}...")
            
            tx = client.get_transaction(tx_hash)
            print(f"   ✓ Transaction retrieved successfully")
            print(f"   From: {tx.get('from', 'Unknown')}")
            print(f"   To: {tx.get('to', 'Unknown')}")
            print(f"   Value: {tx.get('value', 'Unknown')} wei")
            print(f"   Gas: {tx.get('gas', 'Unknown')}")
            print(f"   Gas Price: {tx.get('gasPrice', 'Unknown')} wei")
            
            # Test transaction receipt
            print(f"   Fetching transaction receipt...")
            receipt = client.get_transaction_receipt(tx_hash)
            print(f"   ✓ Transaction receipt retrieved successfully")
            print(f"   Status: {receipt.get('status', 'Unknown')}")
            print(f"   Gas used: {receipt.get('gasUsed', 'Unknown')}")
            
            # Test debug trace if available
            if node_info.get('debug_api_available'):
                print("\n6. Testing debug trace functionality...")
                try:
                    trace = client.debug_trace_transaction(tx_hash)
                    print(f"   ✓ Debug trace retrieved successfully")
                    print(f"   Trace keys: {list(trace.keys()) if isinstance(trace, dict) else 'Non-dict response'}")
                except Exception as e:
                    print(f"   ⚠ Debug trace failed (this is expected for some transactions): {e}")
            else:
                print("\n6. Debug API not available - skipping trace test")
        else:
            print("\n5. No transactions in block - skipping transaction tests")
        
        print("\n=== Test Results ===")
        print("🎉 All basic tests passed! Node connection is working.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the node connection test."""
    setup_logging()
    
    success = test_node_connection()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 