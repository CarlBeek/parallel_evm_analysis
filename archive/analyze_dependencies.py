#!/usr/bin/env python3
"""
Analyze Specific Dependency Chains
Detailed analysis of specific dependency chains found in the continuous collection.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import networkx as nx
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from storage.database import BlockchainDatabase


def get_block_dependency_chains(database: BlockchainDatabase, block_number: int) -> Dict[str, Any]:
    """Get detailed dependency chain information for a specific block."""
    print(f"🔍 Analyzing dependency chains in block {block_number}...")
    
    cursor = database.connection.cursor()
    
    # Get all dependencies for this block
    cursor.execute("""
        SELECT 
            td.dependent_tx_hash,
            td.dependency_tx_hash,
            td.dependency_type,
            td.dependency_reason,
            t1.transaction_index as dependent_index,
            t1.from_address as dependent_from,
            t1.to_address as dependent_to,
            t1.gas_used as dependent_gas,
            t2.transaction_index as dependency_index,
            t2.from_address as dependency_from,
            t2.to_address as dependency_to,
            t2.gas_used as dependency_gas
        FROM transaction_dependencies td
        JOIN transactions t1 ON td.dependent_tx_hash = t1.hash
        JOIN transactions t2 ON td.dependency_tx_hash = t2.hash
        WHERE t1.block_number = ?
        ORDER BY t1.transaction_index
    """, (block_number,))
    
    dependencies = cursor.fetchall()
    
    if not dependencies:
        print(f"❌ No dependencies found for block {block_number}")
        return {}
    
    print(f"✅ Found {len(dependencies)} dependencies in block {block_number}")
    
    # Build dependency graph
    G = nx.DiGraph()
    dependency_details = {}
    
    for dep in dependencies:
        (dependent_tx, dependency_tx, dep_type, dep_reason,
         dependent_idx, dependent_from, dependent_to, dependent_gas,
         dependency_idx, dependency_from, dependency_to, dependency_gas) = dep
        
        G.add_edge(dependency_tx, dependent_tx, 
                  type=dep_type, reason=dep_reason,
                  dependency_index=dependency_idx, dependent_index=dependent_idx)
        
        dependency_details[dependent_tx] = {
            'index': dependent_idx,
            'from': dependent_from,
            'to': dependent_to,
            'gas': dependent_gas
        }
        
        dependency_details[dependency_tx] = {
            'index': dependency_idx,
            'from': dependency_from,
            'to': dependency_to,
            'gas': dependency_gas
        }
    
    # Find all chains
    chains = []
    components = list(nx.weakly_connected_components(G))
    
    for component in components:
        subgraph = G.subgraph(component)
        
        # Find sources and sinks
        sources = [n for n in subgraph.nodes() if subgraph.in_degree(n) == 0]
        sinks = [n for n in subgraph.nodes() if subgraph.out_degree(n) == 0]
        
        if not sources:
            sources = list(subgraph.nodes())[:1]
        if not sinks:
            sinks = list(subgraph.nodes())[-1:]
        
        # Find all paths
        for source in sources:
            for sink in sinks:
                try:
                    if nx.has_path(subgraph, source, sink):
                        path = nx.shortest_path(subgraph, source, sink)
                        if len(path) > 1:  # Only include actual chains
                            chain_info = {
                                'length': len(path),
                                'transactions': path,
                                'source': source,
                                'sink': sink,
                                'total_gas': sum(dependency_details[tx]['gas'] for tx in path),
                                'tx_details': [dependency_details[tx] for tx in path]
                            }
                            chains.append(chain_info)
                except nx.NetworkXNoPath:
                    continue
    
    # Sort chains by length (longest first)
    chains.sort(key=lambda x: x['length'], reverse=True)
    
    return {
        'block_number': block_number,
        'total_dependencies': len(dependencies),
        'total_chains': len(chains),
        'chains': chains,
        'graph': G,
        'dependency_details': dependency_details
    }


def analyze_chain_details(chain_data: Dict[str, Any]) -> None:
    """Print detailed analysis of dependency chains."""
    print(f"\n📊 Dependency Chain Analysis for Block {chain_data['block_number']}")
    print("=" * 80)
    
    print(f"📈 Summary:")
    print(f"   Total dependencies: {chain_data['total_dependencies']}")
    print(f"   Total chains: {chain_data['total_chains']}")
    
    if not chain_data['chains']:
        print("   No chains found")
        return
    
    chains = chain_data['chains']
    
    print(f"   Longest chain: {chains[0]['length']} transactions")
    print(f"   Shortest chain: {chains[-1]['length']} transactions")
    print(f"   Average chain length: {sum(c['length'] for c in chains) / len(chains):.1f}")
    
    # Chain length distribution
    length_dist = {}
    for chain in chains:
        length = chain['length']
        length_dist[length] = length_dist.get(length, 0) + 1
    
    print(f"\n📏 Chain Length Distribution:")
    for length in sorted(length_dist.keys(), reverse=True):
        count = length_dist[length]
        print(f"   Length {length:2d}: {count:3d} chains")
    
    # Show top 5 longest chains in detail
    print(f"\n🔗 Top 5 Longest Chains:")
    for i, chain in enumerate(chains[:5]):
        print(f"\n   Chain {i+1}: {chain['length']} transactions, {chain['total_gas']:,} gas")
        print(f"   Source: {chain['source'][:10]}... (tx {chain['tx_details'][0]['index']})")
        print(f"   Sink: {chain['sink'][:10]}... (tx {chain['tx_details'][-1]['index']})")
        
        # Show transaction sequence
        print(f"   Transaction sequence:")
        for j, (tx_hash, tx_detail) in enumerate(zip(chain['transactions'], chain['tx_details'])):
            print(f"     {j+1:2d}. Tx {tx_detail['index']:3d}: {tx_hash[:10]}... "
                  f"({tx_detail['gas']:,} gas)")
            if tx_detail['to']:
                print(f"         {tx_detail['from'][:10]}... → {tx_detail['to'][:10]}...")


def analyze_longest_chain_block():
    """Analyze the block with the longest dependency chain."""
    database = BlockchainDatabase()
    
    try:
        # From our analysis, we know block 22592433 has the longest chain
        longest_chain_block = 22592433
        
        print(f"🎯 Analyzing block {longest_chain_block} (longest dependency chain)")
        
        chain_data = get_block_dependency_chains(database, longest_chain_block)
        
        if chain_data:
            analyze_chain_details(chain_data)
            
            # Show the actual longest chain in extreme detail
            if chain_data['chains']:
                longest_chain = chain_data['chains'][0]
                print(f"\n" + "=" * 80)
                print(f"🔗 DETAILED ANALYSIS OF LONGEST CHAIN ({longest_chain['length']} transactions)")
                print("=" * 80)
                
                print(f"Total gas consumption: {longest_chain['total_gas']:,}")
                print(f"Average gas per transaction: {longest_chain['total_gas'] / longest_chain['length']:,.0f}")
                
                print(f"\nDetailed transaction flow:")
                for i, (tx_hash, detail) in enumerate(zip(longest_chain['transactions'], longest_chain['tx_details'])):
                    print(f"\n  Step {i+1:2d}: Transaction {detail['index']}")
                    print(f"    Hash: {tx_hash}")
                    print(f"    From: {detail['from']}")
                    print(f"    To:   {detail['to']}")
                    print(f"    Gas:  {detail['gas']:,}")
                    
                    if i < len(longest_chain['transactions']) - 1:
                        print(f"    ↓ depends on ↑")
        
    finally:
        database.close()


def analyze_specific_block(block_number: int):
    """Analyze dependency chains in a specific block."""
    database = BlockchainDatabase()
    
    try:
        chain_data = get_block_dependency_chains(database, block_number)
        
        if chain_data:
            analyze_chain_details(chain_data)
        else:
            print(f"❌ No dependency data found for block {block_number}")
    
    finally:
        database.close()


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze dependency chains in detail')
    parser.add_argument('--block', type=int, help='Specific block number to analyze')
    parser.add_argument('--longest', action='store_true', help='Analyze the block with longest chain')
    
    args = parser.parse_args()
    
    if args.longest:
        analyze_longest_chain_block()
    elif args.block:
        analyze_specific_block(args.block)
    else:
        print("Usage:")
        print("  python analyze_dependencies.py --longest        # Analyze longest chain")
        print("  python analyze_dependencies.py --block 12345    # Analyze specific block")
        
        # Show some summary stats
        database = BlockchainDatabase()
        try:
            stats = database.get_database_stats()
            print(f"\nDatabase contains {stats.get('block_count', 0)} blocks")
            print(f"Block range: {stats.get('block_range', {})}")
            print(f"Total dependencies: {stats.get('dependency_count', 0)}")
        finally:
            database.close()


if __name__ == "__main__":
    main() 