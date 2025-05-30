#!/usr/bin/env python3
"""
Comprehensive debug API test to understand what's available on the Geth node.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.ethereum_client import EthereumClient


def test_debug_api_methods():
    """Test various debug API methods to see what's available."""
    print("=== Comprehensive Debug API Test ===\n")
    
    client = EthereumClient()
    
    # Get a recent transaction to test with
    latest_block = client.get_latest_block_number()
    block = client.w3.eth.get_block(latest_block - 1, full_transactions=True)
    
    if not block.transactions:
        print("No transactions in block, trying another...")
        block = client.w3.eth.get_block(latest_block - 2, full_transactions=True)
    
    if not block.transactions:
        print("❌ No transactions found to test with")
        return
    
    test_tx_hash = block.transactions[0]['hash'].hex()
    print(f"Testing with transaction: {test_tx_hash}")
    print(f"From block: {block.number}\n")
    
    # List of debug methods to test
    debug_methods = [
        'debug_traceTransaction',
        'debug_traceCall',
        'debug_traceBlock',
        'debug_traceBlockByNumber',
        'debug_traceBlockByHash',
        'debug_getBlockRlp',
        'debug_printBlock',
        'debug_dumpBlock',
        'debug_accountRange',
        'debug_storageRangeAt',
        'debug_preimage',
        'debug_getBadBlocks'
    ]
    
    available_methods = []
    unavailable_methods = []
    
    for method in debug_methods:
        print(f"Testing {method}...")
        
        try:
            if method == 'debug_traceTransaction':
                # Test with minimal tracer config
                result = client.w3.manager.request_blocking(method, [test_tx_hash, {"tracer": "callTracer"}])
                available_methods.append(method)
                print(f"  ✅ Available - got result type: {type(result)}")
                
            elif method == 'debug_traceCall':
                # Test with a simple call
                call_params = {
                    'to': block.transactions[0]['to'],
                    'data': '0x',
                    'gas': '0x5208'
                }
                result = client.w3.manager.request_blocking(method, [call_params, 'latest'])
                available_methods.append(method)
                print(f"  ✅ Available - got result type: {type(result)}")
                
            elif method == 'debug_traceBlock':
                # Test with block RLP (this might not work without the RLP)
                try:
                    result = client.w3.manager.request_blocking(method, [f"0x{block.number:x}"])
                    available_methods.append(method)
                    print(f"  ✅ Available - got result type: {type(result)}")
                except Exception as e:
                    if "method not found" in str(e).lower():
                        unavailable_methods.append(method)
                        print(f"  ❌ Method not found")
                    else:
                        available_methods.append(method)
                        print(f"  ⚠️  Available but failed with: {str(e)[:100]}...")
                        
            elif method in ['debug_traceBlockByNumber', 'debug_traceBlockByHash']:
                # Test block tracing
                if method == 'debug_traceBlockByNumber':
                    result = client.w3.manager.request_blocking(method, [f"0x{block.number:x}"])
                else:
                    result = client.w3.manager.request_blocking(method, [block.hash.hex()])
                available_methods.append(method)
                print(f"  ✅ Available - got result type: {type(result)}")
                
            elif method == 'debug_storageRangeAt':
                # Test storage range
                result = client.w3.manager.request_blocking(method, [
                    block.hash.hex(), 0, block.transactions[0]['to'], '0x0', 1
                ])
                available_methods.append(method)
                print(f"  ✅ Available - got result type: {type(result)}")
                
            else:
                # Test other methods with minimal parameters
                result = client.w3.manager.request_blocking(method, [])
                available_methods.append(method)
                print(f"  ✅ Available - got result type: {type(result)}")
                
        except Exception as e:
            error_msg = str(e).lower()
            if "method not found" in error_msg or "does not exist" in error_msg:
                unavailable_methods.append(method)
                print(f"  ❌ Method not found")
            elif "invalid argument" in error_msg:
                available_methods.append(method)
                print(f"  ⚠️  Available but needs different parameters")
            else:
                unavailable_methods.append(method)
                print(f"  ❌ Error: {str(e)[:100]}...")
    
    print(f"\n=== Debug API Test Results ===")
    print(f"✅ Available methods ({len(available_methods)}):")
    for method in available_methods:
        print(f"   • {method}")
    
    print(f"\n❌ Unavailable methods ({len(unavailable_methods)}):")
    for method in unavailable_methods:
        print(f"   • {method}")
    
    print(f"\n📊 Summary:")
    print(f"   Total methods tested: {len(debug_methods)}")
    print(f"   Available: {len(available_methods)}")
    print(f"   Unavailable: {len(unavailable_methods)}")
    print(f"   Availability: {len(available_methods)/len(debug_methods)*100:.1f}%")
    
    # Test the most important method for our use case
    if 'debug_traceTransaction' in available_methods:
        print(f"\n🎉 GOOD NEWS: debug_traceTransaction is available!")
        print(f"   We can get detailed state access information!")
        
        # Test with different tracers
        print(f"\nTesting different tracers...")
        tracers = ['callTracer', 'prestateTracer', 'opcodeTracer', '4byteTracer']
        
        for tracer in tracers:
            try:
                result = client.w3.manager.request_blocking(
                    'debug_traceTransaction', 
                    [test_tx_hash, {"tracer": tracer}]
                )
                print(f"   ✅ {tracer}: Available")
                if tracer == 'prestateTracer':
                    print(f"      This tracer shows state access patterns!")
                    print(f"      Sample result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            except Exception as e:
                print(f"   ❌ {tracer}: {str(e)[:100]}...")
    else:
        print(f"\n⚠️  debug_traceTransaction is NOT available")
        print(f"   This limits our ability to detect true state dependencies")
        print(f"   We may have false negatives (missed dependencies)")
        
        # Check if it's a configuration issue
        print(f"\n🔧 Possible solutions:")
        print(f"   1. Enable debug APIs in Geth with --http.api=eth,net,web3,debug")
        print(f"   2. Check if the node is running with --http.api flag")
        print(f"   3. Verify the node allows debug API access from your IP")


if __name__ == "__main__":
    test_debug_api_methods() 