#!/usr/bin/env python3
"""
Debug script to find complex contract interactions and analyze their storage dependencies.
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.ethereum_client import EthereumClient


def find_complex_transactions():
    """Find complex contract interactions that should have storage dependencies."""
    print("=== Finding Complex Contract Interactions ===\n")
    
    client = EthereumClient()
    
    # Look through several recent blocks to find complex transactions
    latest = client.get_latest_block_number()
    
    for block_offset in range(1, 10):  # Check last 10 blocks
        block_num = latest - block_offset
        block = client.w3.eth.get_block(block_num, full_transactions=True)
        
        print(f"\n--- Block {block_num} ({len(block.transactions)} transactions) ---")
        
        # Find contract interactions with significant input data
        complex_txs = []
        for tx in block.transactions:
            if (tx.get('to') and 
                tx.get('input', '0x') != '0x' and 
                len(tx.get('input', '0x')) > 10 and  # More than just function selector
                tx.get('gas', 0) > 50000):  # Significant gas usage
                complex_txs.append(tx)
        
        print(f"Found {len(complex_txs)} complex contract interactions")
        
        if complex_txs:
            # Analyze the first few complex transactions
            for i, tx in enumerate(complex_txs[:3]):
                tx_hash = tx['hash'].hex()
                print(f"\n  Complex TX {i+1}: {tx_hash}")
                print(f"    To: {tx['to']}")
                print(f"    Gas: {tx['gas']:,}")
                print(f"    Input length: {len(tx['input'])} bytes")
                print(f"    Value: {tx['value']} wei")
                
                # Get transaction receipt to see if it succeeded
                try:
                    receipt = client.w3.eth.get_transaction_receipt(tx_hash)
                    print(f"    Status: {'Success' if receipt['status'] else 'Failed'}")
                    print(f"    Gas used: {receipt['gasUsed']:,}")
                    print(f"    Logs: {len(receipt['logs'])}")
                except Exception as e:
                    print(f"    Receipt error: {e}")
                
                # Trace with prestateTracer
                try:
                    print(f"    Tracing with prestateTracer...")
                    result = client.w3.manager.request_blocking(
                        'debug_traceTransaction', 
                        [tx_hash, {'tracer': 'prestateTracer', 'tracerConfig': {'diffMode': True}}]
                    )
                    
                    if isinstance(result, dict):
                        pre_state = result.get('pre', {})
                        post_state = result.get('post', {})
                        
                        print(f"      Pre-state addresses: {len(pre_state)}")
                        print(f"      Post-state addresses: {len(post_state)}")
                        
                        # Look for storage changes
                        storage_changes = 0
                        for addr in post_state:
                            if addr in pre_state:
                                pre_storage = pre_state[addr].get('storage', {})
                                post_storage = post_state[addr].get('storage', {})
                                
                                if pre_storage or post_storage:
                                    print(f"      {addr}: {len(pre_storage)} -> {len(post_storage)} storage slots")
                                    storage_changes += abs(len(post_storage) - len(pre_storage))
                                    
                                    # Show some storage changes
                                    for slot in post_storage:
                                        pre_val = pre_storage.get(slot, '0x0')
                                        post_val = post_storage[slot]
                                        if pre_val != post_val:
                                            print(f"        Slot {slot}: {pre_val} -> {post_val}")
                        
                        print(f"      Total storage changes: {storage_changes}")
                        
                        # Show full result for first complex transaction
                        if i == 0:
                            print(f"\n      Full prestateTracer result:")
                            print(json.dumps(result, indent=2, default=str)[:3000] + "...")
                    
                except Exception as e:
                    print(f"      Trace error: {e}")
            
            # If we found complex transactions, analyze them for dependencies
            if len(complex_txs) >= 2:
                print(f"\n  Analyzing potential dependencies between complex transactions...")
                
                # Check if any of these transactions interact with the same contracts
                contract_addresses = set()
                for tx in complex_txs:
                    contract_addresses.add(tx['to'])
                
                print(f"  Unique contracts involved: {len(contract_addresses)}")
                print(f"  Contracts: {list(contract_addresses)[:5]}")  # Show first 5
                
                # If multiple transactions hit the same contract, there might be dependencies
                if len(contract_addresses) < len(complex_txs):
                    print(f"  ⚠️  Multiple transactions to same contracts - potential dependencies!")
            
            break  # Found complex transactions, stop searching
    
    else:
        print("No complex contract interactions found in recent blocks")


if __name__ == "__main__":
    find_complex_transactions() 