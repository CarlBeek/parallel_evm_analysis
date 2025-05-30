#!/usr/bin/env python3
"""
Debug a single complex transaction to understand prestateTracer output format.
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.ethereum_client import EthereumClient


def debug_single_transaction():
    """Debug a single complex transaction in detail."""
    print("=== Debugging Single Complex Transaction ===\n")
    
    client = EthereumClient()
    
    # Get a recent block with complex transactions
    latest = client.get_latest_block_number()
    block = client.w3.eth.get_block(latest-1, full_transactions=True)
    
    # Find a complex contract interaction
    complex_tx = None
    for tx in block.transactions:
        if (tx.get('to') and 
            tx.get('input', '0x') != '0x' and 
            len(tx.get('input', '0x')) > 100 and  # Substantial input data
            tx.get('gas', 0) > 100000):  # High gas usage
            complex_tx = tx
            break
    
    if not complex_tx:
        print("No complex transaction found")
        return
    
    tx_hash = complex_tx['hash'].hex()
    print(f"Analyzing transaction: {tx_hash}")
    print(f"To: {complex_tx['to']}")
    print(f"Gas: {complex_tx['gas']:,}")
    print(f"Input length: {len(complex_tx['input'])} bytes")
    
    # Focus on prestateTracer with diffMode
    print(f"\n--- prestateTracer with diffMode ---")
    try:
        result = client.w3.manager.request_blocking(
            'debug_traceTransaction', 
            [tx_hash, {'tracer': 'prestateTracer', 'tracerConfig': {'diffMode': True}}]
        )
        
        print(f"Result type: {type(result)}")
        print(f"Result keys: {list(result.keys()) if hasattr(result, 'keys') else 'No keys method'}")
        
        # Convert AttributeDict to regular dict for easier processing
        if hasattr(result, 'keys'):
            result_dict = dict(result)
            
            print(f"\nConverted to dict, keys: {list(result_dict.keys())}")
            
            if 'pre' in result_dict:
                pre_state = dict(result_dict['pre'])
                print(f"\nPre-state:")
                print(f"  Addresses: {len(pre_state)}")
                
                for i, (addr, data) in enumerate(pre_state.items()):
                    if i >= 3:  # Show first 3 addresses
                        break
                    data_dict = dict(data) if hasattr(data, 'keys') else data
                    print(f"  {addr}:")
                    print(f"    Keys: {list(data_dict.keys()) if isinstance(data_dict, dict) else 'Not a dict'}")
                    
                    if isinstance(data_dict, dict) and 'storage' in data_dict:
                        storage = dict(data_dict['storage']) if hasattr(data_dict['storage'], 'keys') else data_dict['storage']
                        print(f"    Storage slots: {len(storage) if isinstance(storage, dict) else 'Not a dict'}")
                        if isinstance(storage, dict):
                            for j, (slot, value) in enumerate(storage.items()):
                                if j >= 3:  # Show first 3 slots
                                    break
                                print(f"      {slot}: {value}")
            
            if 'post' in result_dict:
                post_state = dict(result_dict['post'])
                print(f"\nPost-state:")
                print(f"  Addresses: {len(post_state)}")
                
                for i, (addr, data) in enumerate(post_state.items()):
                    if i >= 3:  # Show first 3 addresses
                        break
                    data_dict = dict(data) if hasattr(data, 'keys') else data
                    print(f"  {addr}:")
                    print(f"    Keys: {list(data_dict.keys()) if isinstance(data_dict, dict) else 'Not a dict'}")
                    
                    if isinstance(data_dict, dict) and 'storage' in data_dict:
                        storage = dict(data_dict['storage']) if hasattr(data_dict['storage'], 'keys') else data_dict['storage']
                        print(f"    Storage slots: {len(storage) if isinstance(storage, dict) else 'Not a dict'}")
                        if isinstance(storage, dict):
                            for j, (slot, value) in enumerate(storage.items()):
                                if j >= 3:  # Show first 3 slots
                                    break
                                print(f"      {slot}: {value}")
            
            # Analyze storage changes
            if 'pre' in result_dict and 'post' in result_dict:
                print(f"\n--- Storage Change Analysis ---")
                pre_state = dict(result_dict['pre'])
                post_state = dict(result_dict['post'])
                
                total_changes = 0
                addresses_with_changes = 0
                
                # Check all addresses in post state
                for addr in post_state:
                    pre_data = dict(pre_state.get(addr, {})) if addr in pre_state else {}
                    post_data = dict(post_state[addr]) if hasattr(post_state[addr], 'keys') else post_state[addr]
                    
                    if not isinstance(post_data, dict):
                        continue
                    
                    pre_storage = dict(pre_data.get('storage', {})) if 'storage' in pre_data else {}
                    post_storage = dict(post_data.get('storage', {})) if 'storage' in post_data else {}
                    
                    if not isinstance(pre_storage, dict):
                        pre_storage = {}
                    if not isinstance(post_storage, dict):
                        post_storage = {}
                    
                    # Find changes in this address
                    address_changes = 0
                    all_slots = set(pre_storage.keys()) | set(post_storage.keys())
                    
                    for slot in all_slots:
                        pre_val = pre_storage.get(slot, '0x0')
                        post_val = post_storage.get(slot, '0x0')
                        
                        if pre_val != post_val:
                            address_changes += 1
                            total_changes += 1
                            
                            if address_changes <= 3:  # Show first 3 changes per address
                                print(f"  {addr} slot {slot}: {pre_val} -> {post_val}")
                    
                    if address_changes > 0:
                        addresses_with_changes += 1
                        if address_changes > 3:
                            print(f"  {addr}: {address_changes} total changes (showing first 3)")
                
                print(f"\nSummary:")
                print(f"  Addresses with storage changes: {addresses_with_changes}")
                print(f"  Total storage slot changes: {total_changes}")
                
                if total_changes == 0:
                    print(f"  ⚠️  NO STORAGE CHANGES DETECTED!")
                    print(f"  This might explain why we're finding 0 dependencies")
        
        # Show raw result
        print(f"\n--- Raw Result (first 1500 chars) ---")
        try:
            result_str = json.dumps(dict(result), indent=2, default=str)
            print(result_str[:1500] + "..." if len(result_str) > 1500 else result_str)
        except Exception as e:
            print(f"Error serializing result: {e}")
            print(f"Raw result: {str(result)[:1500]}...")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_single_transaction() 