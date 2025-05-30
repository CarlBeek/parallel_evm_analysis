#!/usr/bin/env python3
"""
Multi-block dependency analysis script.
Analyzes 10 blocks to understand dependency patterns across multiple blocks.
"""

import sys
import json
from pathlib import Path
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.ethereum_client import EthereumClient
from core.transaction_fetcher import TransactionFetcher
from storage.database import BlockchainDatabase
from analysis.dependency_analyzer import DependencyAnalyzer
from visualization.dependency_graph import DependencyGraphVisualizer


def analyze_multiple_blocks(num_blocks=10):
    """Analyze multiple blocks for dependency patterns."""
    print(f"=== Multi-Block Dependency Analysis ({num_blocks} blocks) ===\n")
    
    try:
        # Initialize components
        print("1. Initializing components...")
        client = EthereumClient()
        fetcher = TransactionFetcher(client, max_workers=5)  # More workers for faster processing
        database = BlockchainDatabase("./data/multi_block_analysis.db")
        analyzer = DependencyAnalyzer(database)
        visualizer = DependencyGraphVisualizer(database)
        print("   ✓ All components initialized successfully\n")
        
        # Get latest block and calculate range
        latest_block = client.get_latest_block_number()
        start_block = latest_block - num_blocks + 1
        
        print(f"2. Analyzing blocks {start_block} to {latest_block}...")
        
        # Storage for analysis results
        block_results = []
        total_transactions = 0
        total_dependencies = 0
        total_refined_dependencies = 0
        total_independent = 0
        
        # Process each block
        for i, block_num in enumerate(range(start_block, latest_block + 1)):
            print(f"\n   Block {block_num} ({i+1}/{num_blocks}):")
            
            start_time = time.time()
            
            # Fetch block data
            block_data = fetcher.fetch_block_with_transactions(block_num)
            print(f"     Fetched {len(block_data.transactions)} transactions")
            
            # Store in database
            database.store_block(block_data)
            
            # Analyze dependencies
            refined_deps = analyzer.create_refined_dependency_graph(block_data)
            all_deps = database.get_dependencies_for_block(block_num)
            
            # Calculate metrics
            involved_tx_hashes = set()
            for dep in refined_deps:
                involved_tx_hashes.add(dep.dependent_tx_hash)
                involved_tx_hashes.add(dep.dependency_tx_hash)
            
            independent_txs = len(block_data.transactions) - len(involved_tx_hashes)
            parallelization_potential = (independent_txs / len(block_data.transactions)) * 100
            
            # Store results
            block_result = {
                'block_number': block_num,
                'total_transactions': len(block_data.transactions),
                'raw_dependencies': len(all_deps),
                'refined_dependencies': len(refined_deps),
                'independent_transactions': independent_txs,
                'parallelization_potential': parallelization_potential,
                'processing_time': time.time() - start_time
            }
            block_results.append(block_result)
            
            # Update totals
            total_transactions += len(block_data.transactions)
            total_dependencies += len(all_deps)
            total_refined_dependencies += len(refined_deps)
            total_independent += independent_txs
            
            print(f"     Raw deps: {len(all_deps)}, Refined: {len(refined_deps)} ({(1-len(refined_deps)/len(all_deps))*100:.1f}% reduction)")
            print(f"     Parallelization: {parallelization_potential:.1f}% ({independent_txs}/{len(block_data.transactions)})")
            print(f"     Processing time: {time.time() - start_time:.1f}s")
        
        # Generate summary statistics
        print(f"\n3. Generating summary analysis...")
        
        avg_parallelization = (total_independent / total_transactions) * 100
        avg_reduction = (1 - total_refined_dependencies / total_dependencies) * 100
        
        summary = {
            'analysis_summary': {
                'blocks_analyzed': num_blocks,
                'block_range': f"{start_block}-{latest_block}",
                'total_transactions': total_transactions,
                'avg_transactions_per_block': total_transactions / num_blocks,
                'total_raw_dependencies': total_dependencies,
                'total_refined_dependencies': total_refined_dependencies,
                'overall_dependency_reduction': avg_reduction,
                'total_independent_transactions': total_independent,
                'overall_parallelization_potential': avg_parallelization
            },
            'block_details': block_results
        }
        
        # Save detailed results
        results_file = Path("./data/graphs") / f"multi_block_analysis_{start_block}_{latest_block}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"   ✓ Saved detailed results to {results_file}")
        
        # Generate Gantt chart for the most interesting block (highest dependency count)
        most_interesting_block = max(block_results, key=lambda x: x['refined_dependencies'])
        interesting_block_num = most_interesting_block['block_number']
        
        print(f"\n4. Generating Gantt chart for most interesting block ({interesting_block_num})...")
        gantt_fig = visualizer.create_gantt_chart(interesting_block_num, use_refined=True)
        
        gantt_output_file = f"multi_block_gantt_chart_block_{interesting_block_num}"
        visualizer.save_graph(gantt_fig, gantt_output_file, 'html')
        print(f"   ✓ Saved Gantt chart to ./data/graphs/{gantt_output_file}.html")
        
        # Print summary
        print(f"\n=== Multi-Block Analysis Summary ===")
        print(f"📊 Blocks analyzed: {num_blocks} (blocks {start_block}-{latest_block})")
        print(f"📈 Total transactions: {total_transactions:,}")
        print(f"📊 Average transactions per block: {total_transactions/num_blocks:.1f}")
        print(f"🔗 Total raw dependencies: {total_dependencies:,}")
        print(f"✨ Total refined dependencies: {total_refined_dependencies:,}")
        print(f"📉 Overall dependency reduction: {avg_reduction:.1f}%")
        print(f"🚀 Overall parallelization potential: {avg_parallelization:.1f}%")
        print(f"⭐ Most interesting block: {interesting_block_num} ({most_interesting_block['refined_dependencies']} dependencies)")
        
        print(f"\n📁 Generated files:")
        print(f"  - {results_file.name} (detailed analysis)")
        print(f"  - {gantt_output_file}.html (Gantt chart for block {interesting_block_num})")
        
        # Show per-block breakdown
        print(f"\n📋 Per-block breakdown:")
        print(f"{'Block':<8} {'Txs':<5} {'Raw Deps':<8} {'Refined':<8} {'Parallel %':<10} {'Time':<6}")
        print("-" * 55)
        for result in block_results:
            print(f"{result['block_number']:<8} {result['total_transactions']:<5} "
                  f"{result['raw_dependencies']:<8} {result['refined_dependencies']:<8} "
                  f"{result['parallelization_potential']:<9.1f}% {result['processing_time']:<5.1f}s")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if 'database' in locals():
            database.close()


if __name__ == "__main__":
    analyze_multiple_blocks(10) 