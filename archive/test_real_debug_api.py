#!/usr/bin/env python3
"""
Test debug_traceTransaction with real transaction hashes.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.ethereum_client import EthereumClient


def test_real_debug_api():
    """Test debug API with real transaction hashes."""
    print("=== Testing Debug API with Real Transactions ===\n")
    
    client = EthereumClient()
    
    # Get a recent transaction
    latest = client.get_latest_block_number()
    block = client.w3.eth.get_block(latest-1, full_transactions=True)
    
    if not block.transactions:
        print("No transactions found in recent blocks")
        return
    
    tx_hash = block.transactions[0]['hash'].hex()
    print(f"Testing with transaction: {tx_hash}")
    print(f"From block: {block.number}")
    
    # Test different debug methods
    tests = [
        ('debug_traceTransaction with callTracer', 'debug_traceTransaction', [tx_hash, {'tracer': 'callTracer'}]),
        ('debug_traceTransaction with prestateTracer', 'debug_traceTransaction', [tx_hash, {'tracer': 'prestateTracer'}]),
        ('debug_traceTransaction with 4byteTracer', 'debug_traceTransaction', [tx_hash, {'tracer': '4byteTracer'}]),
    ]
    
    for test_name, method, params in tests:
        print(f"\nTesting {test_name}...")
        try:
            result = client.w3.manager.request_blocking(method, params)
            print(f"  ✅ Success! Result type: {type(result)}")
            if isinstance(result, dict):
                print(f"  Keys: {list(result.keys())}")
            elif isinstance(result, list):
                print(f"  List length: {len(result)}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Test the check method
    print(f"\nTesting client.check_debug_api_availability()...")
    try:
        available = client.check_debug_api_availability()
        print(f"  Result: {available}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    test_real_debug_api() 