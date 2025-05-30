#!/usr/bin/env python3
"""
Debug script to examine prestateTracer output and understand why we're not detecting dependencies.
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.ethereum_client import EthereumClient


def debug_prestate_tracer():
    """Debug the prestateTracer output to understand the data format."""
    print("=== Debugging prestateTracer Output ===\n")
    
    client = EthereumClient()
    
    # Get a recent transaction
    latest = client.get_latest_block_number()
    block = client.w3.eth.get_block(latest-1, full_transactions=True)
    
    if not block.transactions:
        print("No transactions found")
        return
    
    # Test multiple transactions to see different patterns
    for i, tx in enumerate(block.transactions[:5]):  # Test first 5 transactions
        tx_hash = tx['hash'].hex()
        print(f"\n--- Transaction {i+1}: {tx_hash} ---")
        print(f"To: {tx.get('to', 'Contract Creation')}")
        print(f"Value: {tx.get('value', 0)} wei")
        print(f"Gas: {tx.get('gas', 0)}")
        
        # Test prestateTracer
        try:
            print(f"\nTesting prestateTracer...")
            result = client.w3.manager.request_blocking(
                'debug_traceTransaction', 
                [tx_hash, {'tracer': 'prestateTracer', 'tracerConfig': {'diffMode': True}}]
            )
            
            print(f"Result type: {type(result)}")
            if isinstance(result, dict):
                print(f"Result keys: {list(result.keys())}")
                
                # Check for pre/post state
                if 'pre' in result:
                    print(f"Pre-state addresses: {len(result['pre'])}")
                    for addr, data in list(result['pre'].items())[:3]:  # Show first 3
                        print(f"  {addr}: {list(data.keys())}")
                        if 'storage' in data:
                            print(f"    Storage slots: {len(data['storage'])}")
                
                if 'post' in result:
                    print(f"Post-state addresses: {len(result['post'])}")
                    for addr, data in list(result['post'].items())[:3]:  # Show first 3
                        print(f"  {addr}: {list(data.keys())}")
                        if 'storage' in data:
                            print(f"    Storage slots: {len(data['storage'])}")
                
                # Show raw result for first transaction
                if i == 0:
                    print(f"\nRaw result preview:")
                    print(json.dumps(result, indent=2, default=str)[:1000] + "...")
            else:
                print(f"Unexpected result format: {result}")
                
        except Exception as e:
            print(f"Error with prestateTracer: {e}")
        
        # Test callTracer for comparison
        try:
            print(f"\nTesting callTracer...")
            call_result = client.w3.manager.request_blocking(
                'debug_traceTransaction', 
                [tx_hash, {'tracer': 'callTracer'}]
            )
            
            print(f"CallTracer result type: {type(call_result)}")
            if isinstance(call_result, dict):
                print(f"CallTracer keys: {list(call_result.keys())}")
                if 'calls' in call_result:
                    print(f"Number of calls: {len(call_result.get('calls', []))}")
                
        except Exception as e:
            print(f"Error with callTracer: {e}")
        
        print(f"\n" + "="*60)
    
    # Test a contract interaction specifically
    print(f"\n--- Looking for Contract Interactions ---")
    contract_txs = [tx for tx in block.transactions if tx.get('to') and tx.get('input', '0x') != '0x']
    
    if contract_txs:
        tx = contract_txs[0]
        tx_hash = tx['hash'].hex()
        print(f"Contract interaction: {tx_hash}")
        print(f"To: {tx['to']}")
        print(f"Input data length: {len(tx.get('input', '0x'))}")
        
        try:
            result = client.w3.manager.request_blocking(
                'debug_traceTransaction', 
                [tx_hash, {'tracer': 'prestateTracer', 'tracerConfig': {'diffMode': True}}]
            )
            
            print(f"\nContract interaction prestateTracer result:")
            print(f"Type: {type(result)}")
            if isinstance(result, dict):
                print(f"Keys: {list(result.keys())}")
                print(json.dumps(result, indent=2, default=str)[:2000] + "...")
                
        except Exception as e:
            print(f"Error tracing contract interaction: {e}")
    else:
        print("No contract interactions found in this block")


if __name__ == "__main__":
    debug_prestate_tracer() 