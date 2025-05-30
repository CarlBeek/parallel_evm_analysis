#!/usr/bin/env python3
"""
Analyze Continuous Collection Data
Analyzes data collected by the continuous collector to show trends and patterns over time.
"""

import sys
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import networkx as nx
from collections import defaultdict, Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from storage.database import BlockchainDatabase
from visualization.dependency_graph import DependencyGraphVisualizer


def analyze_collection_trends(database: BlockchainDatabase) -> Dict[str, Any]:
    """Analyze trends in the continuously collected data."""
    print("📊 Analyzing collection trends...")
    
    # Get all blocks
    cursor = database.connection.cursor()
    cursor.execute("""
        SELECT number, timestamp, gas_used, gas_limit, transaction_count 
        FROM blocks 
        ORDER BY number
    """)
    blocks = [dict(row) for row in cursor.fetchall()]
    
    if not blocks:
        print("❌ No blocks found in database")
        return {}
    
    # Get dependency counts per block
    cursor.execute("""
        SELECT t.block_number, COUNT(td.id) as dependency_count
        FROM blocks b
        LEFT JOIN transactions t ON b.number = t.block_number
        LEFT JOIN transaction_dependencies td ON t.hash = td.dependent_tx_hash
        GROUP BY b.number
        ORDER BY b.number
    """)
    dependency_counts = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Create DataFrame for analysis
    df_data = []
    for block in blocks:
        block_num = block['number']
        df_data.append({
            'block_number': block_num,
            'timestamp': datetime.fromtimestamp(block['timestamp']),
            'gas_used': block['gas_used'],
            'gas_limit': block['gas_limit'],
            'gas_utilization': block['gas_used'] / block['gas_limit'] * 100,
            'transaction_count': block['transaction_count'],
            'dependency_count': dependency_counts.get(block_num, 0),
            'parallelization_potential': (block['transaction_count'] - dependency_counts.get(block_num, 0)) / block['transaction_count'] * 100 if block['transaction_count'] > 0 else 0
        })
    
    df = pd.DataFrame(df_data)
    
    # Calculate statistics
    stats = {
        'total_blocks': len(blocks),
        'total_transactions': df['transaction_count'].sum(),
        'total_dependencies': df['dependency_count'].sum(),
        'avg_transactions_per_block': df['transaction_count'].mean(),
        'avg_dependencies_per_block': df['dependency_count'].mean(),
        'avg_parallelization_potential': df['parallelization_potential'].mean(),
        'block_range': {'min': df['block_number'].min(), 'max': df['block_number'].max()},
        'time_range': {'start': df['timestamp'].min(), 'end': df['timestamp'].max()},
        'collection_duration': df['timestamp'].max() - df['timestamp'].min(),
        'avg_gas_utilization': df['gas_utilization'].mean()
    }
    
    print(f"✅ Analyzed {stats['total_blocks']} blocks:")
    print(f"   📈 Total transactions: {stats['total_transactions']:,}")
    print(f"   🔗 Total dependencies: {stats['total_dependencies']:,}")
    print(f"   ⚡ Avg parallelization: {stats['avg_parallelization_potential']:.1f}%")
    print(f"   ⛽ Avg gas utilization: {stats['avg_gas_utilization']:.1f}%")
    print(f"   📅 Collection duration: {stats['collection_duration']}")
    
    return {'stats': stats, 'dataframe': df}


def create_trend_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create trend visualizations for the continuous data."""
    print("📊 Creating trend visualizations...")
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'Transactions per Block', 'Dependencies per Block',
            'Gas Utilization Over Time', 'Parallelization Potential',
            'Block Processing Timeline', 'Transaction vs Dependency Correlation',
            'Gas Usage Distribution', 'Dependency Rate Over Time'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Transactions per block
    fig.add_trace(
        go.Scatter(x=df['block_number'], y=df['transaction_count'], 
                  mode='lines+markers', name='Transactions',
                  line=dict(color='blue')),
        row=1, col=1
    )
    
    # 2. Dependencies per block
    fig.add_trace(
        go.Scatter(x=df['block_number'], y=df['dependency_count'], 
                  mode='lines+markers', name='Dependencies',
                  line=dict(color='red')),
        row=1, col=2
    )
    
    # 3. Gas utilization over time
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['gas_utilization'], 
                  mode='lines', name='Gas Utilization %',
                  line=dict(color='green')),
        row=2, col=1
    )
    
    # 4. Parallelization potential
    fig.add_trace(
        go.Scatter(x=df['block_number'], y=df['parallelization_potential'], 
                  mode='lines+markers', name='Parallelization %',
                  line=dict(color='purple')),
        row=2, col=2
    )
    
    # 5. Block processing timeline
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['block_number'], 
                  mode='lines+markers', name='Block Timeline',
                  line=dict(color='orange')),
        row=3, col=1
    )
    
    # 6. Transaction vs dependency correlation
    fig.add_trace(
        go.Scatter(x=df['transaction_count'], y=df['dependency_count'], 
                  mode='markers', name='Tx vs Deps',
                  marker=dict(color='cyan', size=8)),
        row=3, col=2
    )
    
    # 7. Gas usage distribution
    fig.add_trace(
        go.Histogram(x=df['gas_used'], name='Gas Usage',
                    marker=dict(color='yellow', opacity=0.7)),
        row=4, col=1
    )
    
    # 8. Dependency rate over time
    df['dependency_rate'] = df['dependency_count'] / df['transaction_count'] * 100
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['dependency_rate'], 
                  mode='lines+markers', name='Dependency Rate %',
                  line=dict(color='magenta')),
        row=4, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1600,
        title_text="Continuous Collection Analysis Dashboard",
        showlegend=False
    )
    
    # Save visualization
    output_file = output_dir / "continuous_collection_trends.html"
    fig.write_html(str(output_file))
    print(f"✅ Saved trend visualization to {output_file}")
    
    return output_file


def analyze_dependency_patterns(database: BlockchainDatabase) -> Dict[str, Any]:
    """Analyze dependency patterns in the collected data."""
    print("🔍 Analyzing dependency patterns...")
    
    cursor = database.connection.cursor()
    
    # Get dependency types distribution
    cursor.execute("""
        SELECT dependency_type, COUNT(*) as count
        FROM transaction_dependencies
        GROUP BY dependency_type
        ORDER BY count DESC
    """)
    dependency_types = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Get top contracts by interactions
    cursor.execute("""
        SELECT contract_address, COUNT(*) as interaction_count
        FROM contract_interactions
        WHERE contract_address != 'pending'
        GROUP BY contract_address
        ORDER BY interaction_count DESC
        LIMIT 10
    """)
    top_contracts = [{'address': row[0], 'interactions': row[1]} for row in cursor.fetchall()]
    
    # Get dependency reasons
    cursor.execute("""
        SELECT dependency_reason, COUNT(*) as count
        FROM transaction_dependencies
        WHERE dependency_reason IS NOT NULL
        GROUP BY dependency_reason
        ORDER BY count DESC
        LIMIT 20
    """)
    dependency_reasons = {row[0]: row[1] for row in cursor.fetchall()}
    
    patterns = {
        'dependency_types': dependency_types,
        'top_contracts': top_contracts,
        'dependency_reasons': dependency_reasons,
        'total_dependencies': sum(dependency_types.values()),
        'unique_contracts': len(top_contracts)
    }
    
    print(f"✅ Dependency analysis complete:")
    print(f"   🔗 Total dependencies: {patterns['total_dependencies']:,}")
    print(f"   📝 Dependency types: {len(dependency_types)}")
    print(f"   🏢 Unique contracts: {patterns['unique_contracts']}")
    
    return patterns


def find_interesting_blocks(database: BlockchainDatabase) -> List[Dict[str, Any]]:
    """Find interesting blocks for detailed analysis."""
    print("🔍 Finding interesting blocks...")
    
    cursor = database.connection.cursor()
    cursor.execute("""
        SELECT 
            b.number,
            b.transaction_count,
            COUNT(td.id) as dependency_count,
            b.gas_used,
            (COUNT(td.id) * 1.0 / b.transaction_count) as dependency_ratio
        FROM blocks b
        LEFT JOIN transactions t ON b.number = t.block_number
        LEFT JOIN transaction_dependencies td ON t.hash = td.dependent_tx_hash
        GROUP BY b.number
        HAVING dependency_count > 0
        ORDER BY dependency_ratio DESC
        LIMIT 10
    """)
    
    interesting_blocks = []
    for row in cursor.fetchall():
        block_num, tx_count, dep_count, gas_used, dep_ratio = row
        interesting_blocks.append({
            'block_number': block_num,
            'transaction_count': tx_count,
            'dependency_count': dep_count,
            'dependency_ratio': dep_ratio,
            'gas_used': gas_used,
            'parallelization_potential': (1 - dep_ratio) * 100
        })
    
    print(f"✅ Found {len(interesting_blocks)} interesting blocks")
    for block in interesting_blocks[:5]:
        print(f"   📦 Block {block['block_number']}: {block['transaction_count']} txs, "
              f"{block['dependency_count']} deps ({block['dependency_ratio']:.1%} ratio)")
    
    return interesting_blocks


def generate_summary_report(database: BlockchainDatabase, output_dir: Path):
    """Generate a comprehensive summary report."""
    print("📊 Generating comprehensive summary report...")
    
    # Get all analysis data
    trends = analyze_collection_trends(database)
    patterns = analyze_dependency_patterns(database)
    interesting_blocks = find_interesting_blocks(database)
    
    # NEW: Add comprehensive dependency chain analysis
    chain_data = analyze_dependency_chains(database)
    temporal_data = analyze_temporal_dependency_patterns(database)
    interesting_chains = find_most_interesting_chains(chain_data, database)
    
    # Create comprehensive report
    report = {
        'generation_time': datetime.now().isoformat(),
        'collection_summary': trends['stats'],
        'dependency_patterns': patterns,
        'interesting_blocks': interesting_blocks,
        'dependency_chains': {
            'chain_statistics': chain_data['chain_analysis'],
            'temporal_patterns': temporal_data['temporal_stats'],
            'interesting_chains': interesting_chains
        },
        'recommendations': {
            'best_blocks_for_analysis': [b['block_number'] for b in interesting_blocks[:3]],
            'best_blocks_for_chains': [
                item['block'] for item in interesting_chains['blocks_with_most_chains'][:3]
            ],
            'longest_chain_block': chain_data['chain_analysis']['longest_chain_block'],
            'avg_parallelization': trends['stats']['avg_parallelization_potential'],
            'collection_health': 'Good' if len(interesting_blocks) > 5 else 'Limited'
        }
    }
    
    # Save comprehensive report
    report_file = output_dir / "continuous_collection_summary.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Saved comprehensive summary report to {report_file}")
    
    # Create all visualizations
    if 'dataframe' in trends:
        create_trend_visualizations(trends['dataframe'], output_dir)
    
    # Create dependency chain visualizations
    create_dependency_chain_visualizations(chain_data, temporal_data, output_dir)
    
    return report


def analyze_dependency_chains(database: BlockchainDatabase) -> Dict[str, Any]:
    """Analyze dependency chains across all collected blocks."""
    print("🔗 Analyzing dependency chains...")
    
    cursor = database.connection.cursor()
    
    # Get all dependencies with block info
    cursor.execute("""
        SELECT 
            td.dependent_tx_hash,
            td.dependency_tx_hash,
            td.dependency_type,
            td.dependency_reason,
            t1.block_number as dependent_block,
            t1.transaction_index as dependent_index,
            t2.transaction_index as dependency_index,
            t1.gas_used as dependent_gas,
            t2.gas_used as dependency_gas
        FROM transaction_dependencies td
        JOIN transactions t1 ON td.dependent_tx_hash = t1.hash
        JOIN transactions t2 ON td.dependency_tx_hash = t2.hash
        ORDER BY t1.block_number, t1.transaction_index
    """)
    
    dependencies = cursor.fetchall()
    
    if not dependencies:
        print("❌ No dependencies found")
        return {}
    
    # Build dependency graphs per block
    block_chains = defaultdict(list)
    block_graphs = defaultdict(lambda: nx.DiGraph())
    chain_lengths = []
    chain_stats = defaultdict(int)
    
    for dep in dependencies:
        (dependent_tx, dependency_tx, dep_type, dep_reason, 
         block_num, dependent_idx, dependency_idx, dependent_gas, dependency_gas) = dep
        
        # Build graph for this block
        G = block_graphs[block_num]
        G.add_edge(dependency_tx, dependent_tx, 
                  type=dep_type, reason=dep_reason,
                  dependency_index=dependency_idx, dependent_index=dependent_idx,
                  dependency_gas=dependency_gas, dependent_gas=dependent_gas)
    
    # Analyze chains in each block
    total_chains = 0
    longest_chain_length = 0
    longest_chain_block = None
    chain_length_distribution = Counter()
    
    for block_num, G in block_graphs.items():
        # Find all simple paths (chains)
        try:
            # Get weakly connected components
            components = list(nx.weakly_connected_components(G))
            
            for component in components:
                subgraph = G.subgraph(component)
                
                # Find sources (nodes with no incoming edges)
                sources = [n for n in subgraph.nodes() if subgraph.in_degree(n) == 0]
                # Find sinks (nodes with no outgoing edges)  
                sinks = [n for n in subgraph.nodes() if subgraph.out_degree(n) == 0]
                
                if not sources:
                    sources = list(subgraph.nodes())[:1]  # Handle cycles
                if not sinks:
                    sinks = list(subgraph.nodes())[-1:]   # Handle cycles
                
                # Find longest paths between sources and sinks
                for source in sources:
                    for sink in sinks:
                        try:
                            if nx.has_path(subgraph, source, sink):
                                path = nx.shortest_path(subgraph, source, sink)
                                chain_length = len(path)
                                chain_lengths.append(chain_length)
                                chain_length_distribution[chain_length] += 1
                                total_chains += 1
                                
                                if chain_length > longest_chain_length:
                                    longest_chain_length = chain_length
                                    longest_chain_block = block_num
                                
                                block_chains[block_num].append({
                                    'length': chain_length,
                                    'transactions': path,
                                    'source': source,
                                    'sink': sink
                                })
                        except nx.NetworkXNoPath:
                            continue
                        except Exception:
                            continue
                            
        except Exception as e:
            print(f"⚠️  Error analyzing block {block_num}: {e}")
            continue
    
    # Calculate statistics
    chain_analysis = {
        'total_chains': total_chains,
        'total_blocks_with_chains': len([b for b in block_chains.keys() if block_chains[b]]),
        'longest_chain_length': longest_chain_length,
        'longest_chain_block': longest_chain_block,
        'avg_chain_length': sum(chain_lengths) / len(chain_lengths) if chain_lengths else 0,
        'chain_length_distribution': dict(chain_length_distribution),
        'block_chain_counts': {block: len(chains) for block, chains in block_chains.items()},
        'dependency_types_in_chains': {},
        'gas_impact_by_chain_length': defaultdict(list)
    }
    
    # Analyze dependency types in chains
    dep_types_counter = Counter()
    for dep in dependencies:
        dep_types_counter[dep[2]] += 1
    chain_analysis['dependency_types_in_chains'] = dict(dep_types_counter)
    
    # Analyze gas impact by chain length
    for block_num, chains in block_chains.items():
        for chain in chains:
            G = block_graphs[block_num]
            total_gas = 0
            for tx in chain['transactions']:
                if G.has_node(tx):
                    # Sum gas from all edges involving this transaction
                    for _, _, data in G.edges(tx, data=True):
                        total_gas += data.get('dependent_gas', 0)
            
            chain_analysis['gas_impact_by_chain_length'][chain['length']].append(total_gas)
    
    # Calculate averages for gas impact
    for length in chain_analysis['gas_impact_by_chain_length']:
        gas_values = chain_analysis['gas_impact_by_chain_length'][length]
        chain_analysis['gas_impact_by_chain_length'][length] = {
            'avg_gas': sum(gas_values) / len(gas_values) if gas_values else 0,
            'total_gas': sum(gas_values),
            'count': len(gas_values)
        }
    
    print(f"✅ Chain analysis complete:")
    print(f"   🔗 Total chains found: {total_chains:,}")
    print(f"   📦 Blocks with chains: {chain_analysis['total_blocks_with_chains']}")
    print(f"   📏 Longest chain: {longest_chain_length} transactions (block {longest_chain_block})")
    print(f"   📊 Average chain length: {chain_analysis['avg_chain_length']:.1f}")
    print(f"   📈 Chain length distribution: {dict(list(chain_length_distribution.most_common(5)))}")
    
    return {
        'chain_analysis': chain_analysis,
        'block_chains': dict(block_chains),
        'block_graphs': block_graphs
    }


def analyze_temporal_dependency_patterns(database: BlockchainDatabase) -> Dict[str, Any]:
    """Analyze how dependency patterns change over time."""
    print("⏰ Analyzing temporal dependency patterns...")
    
    cursor = database.connection.cursor()
    
    # Get dependency counts and characteristics over time
    cursor.execute("""
        SELECT 
            b.number as block_number,
            b.timestamp,
            b.transaction_count,
            COUNT(td.id) as dependency_count,
            AVG(CASE WHEN td.gas_impact IS NOT NULL THEN td.gas_impact ELSE 0 END) as avg_gas_impact
        FROM blocks b
        LEFT JOIN transactions t ON b.number = t.block_number
        LEFT JOIN transaction_dependencies td ON t.hash = td.dependent_tx_hash
        GROUP BY b.number, b.timestamp, b.transaction_count
        ORDER BY b.number
    """)
    
    temporal_data = []
    for row in cursor.fetchall():
        block_num, timestamp, tx_count, dep_count, avg_gas = row
        temporal_data.append({
            'block_number': block_num,
            'timestamp': datetime.fromtimestamp(timestamp),
            'transaction_count': tx_count,
            'dependency_count': dep_count or 0,
            'dependency_rate': (dep_count or 0) / tx_count * 100 if tx_count > 0 else 0,
            'avg_gas_impact': avg_gas or 0
        })
    
    # Calculate moving averages
    df = pd.DataFrame(temporal_data)
    if len(df) > 5:
        df['dependency_rate_ma5'] = df['dependency_rate'].rolling(window=5, center=True).mean()
        df['transaction_count_ma5'] = df['transaction_count'].rolling(window=5, center=True).mean()
    
    temporal_stats = {
        'total_timespan': df['timestamp'].max() - df['timestamp'].min() if len(df) > 1 else None,
        'avg_dependency_rate': df['dependency_rate'].mean(),
        'dependency_rate_trend': df['dependency_rate'].diff().mean(),  # Positive = increasing
        'peak_dependency_block': df.loc[df['dependency_rate'].idxmax(), 'block_number'] if len(df) > 0 else None,
        'peak_dependency_rate': df['dependency_rate'].max(),
        'temporal_correlation': df['dependency_rate'].corr(df['transaction_count']) if len(df) > 1 else 0
    }
    
    print(f"✅ Temporal analysis complete:")
    print(f"   ⏱️  Timespan: {temporal_stats['total_timespan']}")
    print(f"   📊 Avg dependency rate: {temporal_stats['avg_dependency_rate']:.1f}%")
    print(f"   📈 Dependency trend: {temporal_stats['dependency_rate_trend']:.3f}% per block")
    print(f"   🎯 Peak dependency: {temporal_stats['peak_dependency_rate']:.1f}% (block {temporal_stats['peak_dependency_block']})")
    
    return {
        'temporal_stats': temporal_stats,
        'temporal_dataframe': df
    }


def create_dependency_chain_visualizations(chain_data: Dict, temporal_data: Dict, output_dir: Path):
    """Create comprehensive dependency chain visualizations."""
    print("📊 Creating dependency chain visualizations...")
    
    chain_analysis = chain_data['chain_analysis']
    temporal_df = temporal_data['temporal_dataframe']
    
    # Create comprehensive chain analysis dashboard
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'Chain Length Distribution', 'Chains per Block',
            'Dependency Rate Over Time', 'Gas Impact by Chain Length',
            'Chain Count vs Transaction Count', 'Temporal Dependency Trend',
            'Dependency Types Distribution', 'Chain Characteristics'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Chain Length Distribution
    chain_lengths = list(chain_analysis['chain_length_distribution'].keys())
    chain_counts = list(chain_analysis['chain_length_distribution'].values())
    
    fig.add_trace(
        go.Bar(x=chain_lengths, y=chain_counts, name='Chain Lengths',
               marker=dict(color='skyblue')),
        row=1, col=1
    )
    
    # 2. Chains per Block
    block_numbers = list(chain_analysis['block_chain_counts'].keys())
    chains_per_block = list(chain_analysis['block_chain_counts'].values())
    
    fig.add_trace(
        go.Scatter(x=block_numbers, y=chains_per_block, 
                  mode='lines+markers', name='Chains per Block',
                  line=dict(color='red')),
        row=1, col=2
    )
    
    # 3. Dependency Rate Over Time
    fig.add_trace(
        go.Scatter(x=temporal_df['timestamp'], y=temporal_df['dependency_rate'], 
                  mode='lines', name='Dependency Rate %',
                  line=dict(color='green')),
        row=2, col=1
    )
    
    # Add moving average if available
    if 'dependency_rate_ma5' in temporal_df.columns:
        fig.add_trace(
            go.Scatter(x=temporal_df['timestamp'], y=temporal_df['dependency_rate_ma5'], 
                      mode='lines', name='5-Block Moving Avg',
                      line=dict(color='darkgreen', dash='dash')),
            row=2, col=1
        )
    
    # 4. Gas Impact by Chain Length
    gas_impact_data = chain_analysis['gas_impact_by_chain_length']
    if gas_impact_data:
        lengths = list(gas_impact_data.keys())
        avg_gas = [gas_impact_data[l]['avg_gas'] for l in lengths]
        
        fig.add_trace(
            go.Bar(x=lengths, y=avg_gas, name='Avg Gas Impact',
                   marker=dict(color='orange')),
            row=2, col=2
        )
    
    # 5. Chain Count vs Transaction Count
    fig.add_trace(
        go.Scatter(x=temporal_df['transaction_count'], y=temporal_df['dependency_count'], 
                  mode='markers', name='Chains vs Transactions',
                  marker=dict(color='purple', size=8)),
        row=3, col=1
    )
    
    # 6. Temporal Dependency Trend
    if len(temporal_df) > 1:
        fig.add_trace(
            go.Scatter(x=temporal_df['block_number'], y=temporal_df['dependency_rate'], 
                      mode='lines+markers', name='Dependency Trend',
                      line=dict(color='brown')),
            row=3, col=2
        )
    
    # 7. Dependency Types Distribution
    dep_types = list(chain_analysis['dependency_types_in_chains'].keys())
    dep_type_counts = list(chain_analysis['dependency_types_in_chains'].values())
    
    if dep_types:
        fig.add_trace(
            go.Bar(x=dep_types, y=dep_type_counts, name='Dependency Types',
                   marker=dict(color='cyan')),
            row=4, col=1
        )
    
    # 8. Chain Characteristics Summary
    # Create a text summary
    summary_text = [
        f"Total Chains: {chain_analysis['total_chains']:,}",
        f"Blocks with Chains: {chain_analysis['total_blocks_with_chains']}",
        f"Longest Chain: {chain_analysis['longest_chain_length']} txs",
        f"Avg Chain Length: {chain_analysis['avg_chain_length']:.1f}",
        f"Peak Dependency Rate: {temporal_data['temporal_stats']['peak_dependency_rate']:.1f}%"
    ]
    
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='text',
                  text=['<br>'.join(summary_text)],
                  textposition='middle center',
                  showlegend=False),
        row=4, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1800,
        title_text="Dependency Chain Analysis Dashboard",
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Chain Length", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    
    fig.update_xaxes(title_text="Block Number", row=1, col=2)
    fig.update_yaxes(title_text="Chain Count", row=1, col=2)
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Dependency Rate %", row=2, col=1)
    
    fig.update_xaxes(title_text="Chain Length", row=2, col=2)
    fig.update_yaxes(title_text="Avg Gas Impact", row=2, col=2)
    
    fig.update_xaxes(title_text="Transaction Count", row=3, col=1)
    fig.update_yaxes(title_text="Dependency Count", row=3, col=1)
    
    fig.update_xaxes(title_text="Block Number", row=3, col=2)
    fig.update_yaxes(title_text="Dependency Rate %", row=3, col=2)
    
    # Save visualization
    output_file = output_dir / "dependency_chain_analysis.html"
    fig.write_html(str(output_file))
    print(f"✅ Saved chain analysis dashboard to {output_file}")
    
    return output_file


def find_most_interesting_chains(chain_data: Dict, database: BlockchainDatabase) -> Dict[str, Any]:
    """Find the most interesting dependency chains for detailed analysis."""
    print("🔍 Finding most interesting dependency chains...")
    
    block_chains = chain_data['block_chains']
    chain_analysis = chain_data['chain_analysis']
    
    # Find interesting chains
    interesting_chains = []
    
    # 1. Longest chains
    longest_chains = []
    max_length = chain_analysis['longest_chain_length']
    
    for block_num, chains in block_chains.items():
        for chain in chains:
            if chain['length'] == max_length:
                longest_chains.append({
                    'block': block_num,
                    'length': chain['length'],
                    'transactions': chain['transactions'],
                    'type': 'longest_chain'
                })
    
    # 2. Blocks with most chains
    blocks_by_chain_count = sorted(
        chain_analysis['block_chain_counts'].items(),
        key=lambda x: x[1], reverse=True
    )[:5]
    
    # 3. Get transaction details for interesting chains
    cursor = database.connection.cursor()
    
    interesting_patterns = {
        'longest_chains': longest_chains,
        'blocks_with_most_chains': [
            {'block': block, 'chain_count': count} 
            for block, count in blocks_by_chain_count
        ],
        'chain_examples': []
    }
    
    # Get detailed info for a few example chains
    for block_num, chains in list(block_chains.items())[:3]:
        if chains:
            example_chain = max(chains, key=lambda x: x['length'])
            
            # Get transaction details
            tx_hashes = "', '".join(example_chain['transactions'])
            cursor.execute(f"""
                SELECT hash, transaction_index, from_address, to_address, gas_used
                FROM transactions 
                WHERE hash IN ('{tx_hashes}')
                ORDER BY transaction_index
            """)
            
            tx_details = [dict(row) for row in cursor.fetchall()]
            
            interesting_patterns['chain_examples'].append({
                'block': block_num,
                'chain_length': example_chain['length'],
                'transactions': tx_details
            })
    
    print(f"✅ Found interesting patterns:")
    print(f"   📏 Longest chains: {len(longest_chains)} chains of length {max_length}")
    print(f"   📦 Top blocks by chain count: {blocks_by_chain_count[:3]}")
    
    return interesting_patterns


def main():
    """Main analysis function."""
    print("🔄 Comprehensive Continuous Collection Data Analysis")
    print("=" * 60)
    
    # Initialize database
    database = BlockchainDatabase()
    
    # Check if we have data
    stats = database.get_database_stats()
    if stats.get('block_count', 0) == 0:
        print("❌ No data found in database. Run continuous collection first!")
        return 1
    
    print(f"📊 Found {stats.get('block_count', 0)} blocks in database")
    
    # Create output directory
    output_dir = Path("data/continuous_analysis")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Generate comprehensive analysis
        report = generate_summary_report(database, output_dir)
        
        print("\n" + "=" * 60)
        print("📈 COMPREHENSIVE ANALYSIS COMPLETE")
        print("=" * 60)
        
        print(f"📊 Collection Summary:")
        summary = report['collection_summary']
        print(f"   Blocks analyzed: {summary['total_blocks']}")
        print(f"   Total transactions: {summary['total_transactions']:,}")
        print(f"   Total dependencies: {summary['total_dependencies']:,}")
        print(f"   Avg parallelization: {summary['avg_parallelization_potential']:.1f}%")
        print(f"   Collection duration: {summary['collection_duration']}")
        
        print(f"\n🔗 Dependency Chain Analysis:")
        chains = report['dependency_chains']['chain_statistics']
        print(f"   Total chains found: {chains['total_chains']:,}")
        print(f"   Blocks with chains: {chains['total_blocks_with_chains']}")
        print(f"   Longest chain: {chains['longest_chain_length']} transactions")
        print(f"   Average chain length: {chains['avg_chain_length']:.1f}")
        print(f"   Chain length distribution: {chains['chain_length_distribution']}")
        
        print(f"\n⏰ Temporal Patterns:")
        temporal = report['dependency_chains']['temporal_patterns']
        print(f"   Dependency rate trend: {temporal['dependency_rate_trend']:.3f}% per block")
        print(f"   Peak dependency rate: {temporal['peak_dependency_rate']:.1f}%")
        print(f"   Peak dependency block: {temporal['peak_dependency_block']}")
        
        print(f"\n🎯 Recommendations:")
        recs = report['recommendations']
        print(f"   Best blocks for analysis: {recs['best_blocks_for_analysis']}")
        print(f"   Best blocks for chain analysis: {recs['best_blocks_for_chains']}")
        print(f"   Longest chain block: {recs['longest_chain_block']}")
        print(f"   Overall parallelization: {recs['avg_parallelization']:.1f}%")
        print(f"   Collection health: {recs['collection_health']}")
        
        print(f"\n📁 Output files in {output_dir}:")
        print(f"   - continuous_collection_summary.json (comprehensive report)")
        print(f"   - continuous_collection_trends.html (general trends dashboard)")
        print(f"   - dependency_chain_analysis.html (chain analysis dashboard)")
        
        print(f"\n🚀 Try analyzing specific blocks:")
        for block_num in recs['best_blocks_for_analysis'][:3]:
            print(f"   python demo_clean_visualization.py --block {block_num}")
        
        print(f"\n🔗 Best blocks for dependency chain analysis:")
        for block_num in recs['best_blocks_for_chains'][:3]:
            print(f"   python demo_clean_visualization.py --block {block_num}")
        
        print(f"\n📏 Longest dependency chain:")
        print(f"   python demo_clean_visualization.py --block {recs['longest_chain_block']}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        database.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 