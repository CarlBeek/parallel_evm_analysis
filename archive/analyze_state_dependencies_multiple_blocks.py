#!/usr/bin/env python3
"""
Analyze state dependencies across multiple blocks using exact state access tracking.
"""

import sys
import json
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.ethereum_client import EthereumClient
from core.transaction_fetcher import TransactionFetcher
from storage.database import BlockchainDatabase
from analysis.state_dependency_analyzer import StateDependencyAnalyzer


def analyze_state_dependencies_multiple_blocks(num_blocks=5):
    """Analyze state dependencies across multiple blocks."""
    print(f"=== State Dependency Analysis ({num_blocks} blocks) ===\n")
    
    try:
        # Initialize components
        print("1. Initializing components...")
        client = EthereumClient()
        fetcher = TransactionFetcher(client)
        database = BlockchainDatabase("./data/state_dependency_analysis.db")
        
        if not client.check_debug_api_availability():
            print("❌ Debug API not available")
            return False
        
        state_analyzer = StateDependencyAnalyzer(client, database)
        print("   ✅ All components initialized")
        
        # Get blocks to analyze
        latest_block = client.get_latest_block_number()
        start_block = latest_block - num_blocks - 1
        end_block = latest_block - 2
        
        print(f"\n2. Analyzing blocks {start_block} to {end_block}...")
        
        results = []
        total_start_time = time.time()
        
        for i, block_num in enumerate(range(start_block, end_block + 1)):
            block_start_time = time.time()
            print(f"\n--- Block {block_num} ({i+1}/{num_blocks}) ---")
            
            # Fetch block
            block_data = fetcher.fetch_block_with_transactions(block_num)
            print(f"   Transactions: {len(block_data.transactions)}")
            
            # Store block
            database.store_block(block_data)
            
            # State-based analysis
            print(f"   Running state-based dependency analysis...")
            state_deps = state_analyzer.analyze_block_state_dependencies(block_data)
            
            # Calculate total gas for this block
            total_gas = sum(tx.gas_used or 0 for tx in block_data.transactions)
            
            # Get analysis
            analysis = state_analyzer.get_parallelization_analysis(
                state_deps, len(block_data.transactions), total_gas
            )
            
            block_time = time.time() - block_start_time
            
            result = {
                'block_number': block_num,
                'total_transactions': len(block_data.transactions),
                'total_gas': total_gas,
                'state_dependencies': len(state_deps),
                'dependent_gas': analysis['dependent_gas'],
                'independent_gas': analysis['independent_gas'],
                'gas_parallelization_percent': analysis['gas_parallelization_potential_percent'],
                'gas_theoretical_speedup': analysis['gas_theoretical_speedup'],
                'tx_parallelization_percent': analysis['tx_parallelization_potential_percent'],
                'tx_theoretical_speedup': analysis['tx_theoretical_speedup'],
                'dependency_chains': analysis['dependency_chains'],
                'longest_chain': analysis['longest_dependency_chain'],
                'critical_path_gas': analysis['critical_path_gas'],
                'analysis_time_seconds': block_time
            }
            
            results.append(result)
            
            print(f"   Results:")
            print(f"     State dependencies: {len(state_deps)}")
            print(f"     Gas parallelization potential: {analysis['gas_parallelization_potential_percent']:.1f}%")
            print(f"     Gas theoretical speedup: {analysis['gas_theoretical_speedup']:.1f}x")
            print(f"     TX parallelization potential: {analysis['tx_parallelization_potential_percent']:.1f}%")
            print(f"     TX theoretical speedup: {analysis['tx_theoretical_speedup']:.1f}x")
            print(f"     Dependency chains: {analysis['dependency_chains']}")
            print(f"     Longest chain: {analysis['longest_dependency_chain']}")
            print(f"     Critical path gas: {analysis['critical_path_gas']:,}")
            print(f"     Analysis time: {block_time:.1f}s")
            
            # Show some example dependencies
            if state_deps:
                print(f"   Example dependencies:")
                for j, dep in enumerate(state_deps[:3]):  # Show first 3
                    print(f"     {j+1}. TX {dep.dependency_tx_index} -> TX {dep.dependent_tx_index}")
                    print(f"        Contract: {dep.contract_address}")
                    print(f"        Storage slot: {dep.storage_slot}")
                    print(f"        Reason: {dep.dependency_reason}")
                    print(f"        Gas impact: {dep.gas_impact:,}")
                if len(state_deps) > 3:
                    print(f"     ... and {len(state_deps) - 3} more dependencies")
        
        total_time = time.time() - total_start_time
        print(f"\n3. Analysis complete! Total time: {total_time:.1f}s")
        
        # Summary statistics
        print(f"\n=== Summary Statistics ===")
        total_transactions = sum(r['total_transactions'] for r in results)
        total_dependencies = sum(r['state_dependencies'] for r in results)
        avg_parallelization = sum(r['gas_parallelization_percent'] for r in results) / len(results)
        avg_speedup = sum(r['gas_theoretical_speedup'] for r in results) / len(results)
        
        print(f"Blocks analyzed: {len(results)}")
        print(f"Total transactions: {total_transactions:,}")
        print(f"Total state dependencies: {total_dependencies:,}")
        print(f"Dependency rate: {(total_dependencies/total_transactions)*100:.2f}% of transactions")
        print(f"Average gas parallelization potential: {avg_parallelization:.1f}%")
        print(f"Average gas theoretical speedup: {avg_speedup:.1f}x")
        print(f"Average analysis time per block: {total_time/len(results):.1f}s")
        
        # Save results
        output_file = f"data/state_dependency_analysis_{start_block}_{end_block}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'analysis_metadata': {
                    'start_block': start_block,
                    'end_block': end_block,
                    'blocks_analyzed': len(results),
                    'total_analysis_time_seconds': total_time,
                    'analysis_method': 'exact_state_access_tracking'
                },
                'summary': {
                    'total_transactions': total_transactions,
                    'total_dependencies': total_dependencies,
                    'dependency_rate_percent': (total_dependencies/total_transactions)*100,
                    'avg_gas_parallelization_percent': avg_parallelization,
                    'avg_gas_theoretical_speedup': avg_speedup
                },
                'block_results': results
            }, f, indent=2)
        
        print(f"\n📊 Results saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'database' in locals():
            database.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='State Dependency Analysis')
    parser.add_argument('--blocks', type=int, default=5, help='Number of blocks to analyze (default: 5)')
    
    args = parser.parse_args()
    
    analyze_state_dependencies_multiple_blocks(args.blocks) 