#!/usr/bin/env python3
"""
Comprehensive Ethereum Parallelization Analysis
Analyzes multiple blocks to understand parallelization patterns and generates detailed statistics and visualizations.
OPTIMIZED VERSION - Uses parallel processing and caching for speed.
"""

import sys
import json
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.ethereum_client import EthereumClient
from core.transaction_fetcher import TransactionFetcher
from storage.database import BlockchainDatabase
from analysis.state_dependency_analyzer import StateDependencyAnalyzer


def analyze_ethereum_parallelization(num_blocks=20, max_workers=8):
    """Comprehensive analysis of Ethereum parallelization across multiple blocks."""
    print(f"=== Comprehensive Ethereum Parallelization Analysis ({num_blocks} blocks) ===")
    print(f"Using {max_workers} parallel workers for optimization\n")
    
    try:
        # Initialize components
        print("1. Initializing components...")
        client = EthereumClient()
        fetcher = TransactionFetcher(client)
        database = BlockchainDatabase("./data/comprehensive_parallelization_analysis.db")
        
        if not client.check_debug_api_availability():
            print("❌ Debug API not available")
            return False
        
        # Use optimized analyzer with parallel processing
        state_analyzer = StateDependencyAnalyzer(client, database, max_workers=max_workers)
        print("   ✅ All components initialized")
        
        # Get blocks to analyze
        latest_block = client.get_latest_block_number()
        start_block = latest_block - num_blocks - 1
        end_block = latest_block - 2
        
        print(f"\n2. Analyzing blocks {start_block} to {end_block}...")
        print(f"   Progress will be shown for each block...")
        
        results = []
        start_time = time.time()
        
        for i, block_num in enumerate(range(start_block, end_block + 1)):
            block_start_time = time.time()
            progress = f"({i+1}/{num_blocks})"
            print(f"\n--- Block {block_num} {progress} ---")
            
            # Fetch block
            fetch_start = time.time()
            block_data = fetcher.fetch_block_with_transactions(block_num)
            fetch_time = time.time() - fetch_start
            print(f"   📦 Fetched {len(block_data.transactions)} transactions in {fetch_time:.1f}s")
            
            # Store block
            store_start = time.time()
            database.store_block(block_data)
            store_time = time.time() - store_start
            print(f"   💾 Stored block data in {store_time:.1f}s")
            
            # State-based analysis (this is the slow part, now optimized)
            print(f"   🔍 Running parallel state-based analysis...")
            analysis_start = time.time()
            state_deps = state_analyzer.analyze_block_state_dependencies(block_data)
            
            # Calculate total gas for this block
            total_gas = sum(tx.gas_used or 0 for tx in block_data.transactions)
            
            state_analysis = state_analyzer.get_parallelization_analysis(
                state_deps, len(block_data.transactions), total_gas
            )
            analysis_time = time.time() - analysis_start
            
            # Calculate additional metrics
            dependent_gas = state_analysis['dependent_gas']
            independent_gas = state_analysis['independent_gas']
            
            # Find largest dependency by gas
            largest_dep_gas = max((dep.gas_impact for dep in state_deps), default=0)
            
            # Calculate dependency chain lengths and gas amounts
            chain_gas_amounts = state_analysis.get('chain_gas_amounts', [])
            chain_lengths = state_analysis.get('chain_lengths', [])
            avg_chain_length = state_analysis.get('avg_chain_length', 0)
            max_chain_length = state_analysis.get('longest_dependency_chain', 0)
            avg_chain_gas = state_analysis.get('avg_chain_gas', 0)
            max_chain_gas = state_analysis.get('longest_chain_gas', 0)
            
            # Block timing
            block_time = time.time() - block_start_time
            
            result = {
                'block_number': block_num,
                'timestamp': block_data.timestamp,
                'total_transactions': len(block_data.transactions),
                'state_dependencies': len(state_deps),
                'dependent_transactions': state_analysis['dependent_transactions'],
                'independent_transactions': state_analysis['independent_transactions'],
                
                # Gas-based metrics (primary)
                'total_gas': total_gas,
                'dependent_gas': dependent_gas,
                'independent_gas': independent_gas,
                'gas_parallelization_percent': state_analysis['gas_parallelization_potential_percent'],
                'gas_theoretical_speedup': state_analysis['gas_theoretical_speedup'],
                'critical_path_gas': state_analysis['critical_path_gas'],
                
                # Transaction-based metrics (for comparison)
                'tx_parallelization_percent': state_analysis['tx_parallelization_potential_percent'],
                'tx_theoretical_speedup': state_analysis['tx_theoretical_speedup'],
                
                # Chain analysis (gas-based)
                'dependency_chains': state_analysis['dependency_chains'],
                'avg_chain_length': avg_chain_length,
                'max_chain_length': max_chain_length,
                'avg_chain_gas': avg_chain_gas,
                'max_chain_gas': max_chain_gas,
                'total_chain_gas': state_analysis.get('total_chain_gas', 0),
                'chain_gas_amounts': chain_gas_amounts,
                'chain_lengths': chain_lengths,
                
                'largest_dependency_gas': largest_dep_gas,
                'gas_utilization': (total_gas / block_data.gas_limit) * 100,
                'analysis_time_seconds': analysis_time,
                'total_block_time_seconds': block_time
            }
            
            results.append(result)
            
            # Progress summary
            elapsed = time.time() - start_time
            avg_time_per_block = elapsed / (i + 1)
            estimated_remaining = avg_time_per_block * (num_blocks - i - 1)
            
            print(f"   ✅ Results: {len(state_deps)} deps, {result['gas_parallelization_percent']:.1f}% gas parallel, {result['gas_theoretical_speedup']:.1f}x gas speedup")
            print(f"   📊 TX-based: {result['tx_parallelization_percent']:.1f}% parallel, {result['tx_theoretical_speedup']:.1f}x speedup")
            print(f"   ⏱️  Block time: {block_time:.1f}s (analysis: {analysis_time:.1f}s)")
            print(f"   📈 Progress: {((i+1)/num_blocks)*100:.1f}% complete, ~{estimated_remaining/60:.1f}min remaining")
        
        total_time = time.time() - start_time
        print(f"\n3. Analysis complete! Total time: {total_time/60:.1f} minutes")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Generate comprehensive statistics
        print(f"\n=== Comprehensive Statistics ===")
        generate_statistics(df)
        
        # Generate visualizations
        print(f"\n4. Generating visualizations...")
        generate_visualizations(df, start_block, end_block)
        
        # Save detailed results
        output_file = f"data/comprehensive_analysis_{start_block}_{end_block}.json"
        detailed_results = {
            'analysis_metadata': {
                'start_block': start_block,
                'end_block': end_block,
                'blocks_analyzed': len(results),
                'total_analysis_time_seconds': total_time,
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_method': 'exact_state_access_tracking_parallel',
                'max_workers': max_workers,
                'avg_time_per_block_seconds': total_time / len(results)
            },
            'summary_statistics': calculate_summary_statistics(df),
            'block_results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"📊 Detailed results saved to: {output_file}")
        print(f"📈 Visualizations saved to: data/graphs/")
        print(f"⚡ Average analysis time per block: {total_time/len(results):.1f}s")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'database' in locals():
            database.close()


def generate_statistics(df):
    """Generate comprehensive statistics from the analysis results."""
    total_blocks = len(df)
    total_transactions = df['total_transactions'].sum()
    total_dependencies = df['state_dependencies'].sum()
    total_gas = df['total_gas'].sum()
    total_dependent_gas = df['dependent_gas'].sum()
    total_independent_gas = df['independent_gas'].sum()
    
    print(f"Blocks analyzed: {total_blocks}")
    print(f"Total transactions: {total_transactions:,}")
    print(f"Total gas analyzed: {total_gas:,}")
    print(f"Total state dependencies: {total_dependencies:,}")
    print(f"Average dependencies per block: {total_dependencies/total_blocks:.1f}")
    print(f"Dependency rate: {(total_dependencies/total_transactions)*100:.2f}% of transactions")
    
    print(f"\n--- Gas-Based Parallelization Metrics (Primary) ---")
    print(f"Total dependent gas: {total_dependent_gas:,} ({(total_dependent_gas/total_gas)*100:.1f}%)")
    print(f"Total independent gas: {total_independent_gas:,} ({(total_independent_gas/total_gas)*100:.1f}%)")
    print(f"Average gas parallelization potential: {df['gas_parallelization_percent'].mean():.1f}%")
    print(f"Min gas parallelization potential: {df['gas_parallelization_percent'].min():.1f}%")
    print(f"Max gas parallelization potential: {df['gas_parallelization_percent'].max():.1f}%")
    print(f"Std dev gas parallelization: {df['gas_parallelization_percent'].std():.1f}%")
    
    print(f"\n--- Gas-Based Speedup Metrics (Primary) ---")
    print(f"Average gas theoretical speedup: {df['gas_theoretical_speedup'].mean():.1f}x")
    print(f"Min gas theoretical speedup: {df['gas_theoretical_speedup'].min():.1f}x")
    print(f"Max gas theoretical speedup: {df['gas_theoretical_speedup'].max():.1f}x")
    print(f"Median gas theoretical speedup: {df['gas_theoretical_speedup'].median():.1f}x")
    
    print(f"\n--- Transaction-Based Metrics (For Comparison) ---")
    print(f"Average tx parallelization potential: {df['tx_parallelization_percent'].mean():.1f}%")
    print(f"Average tx theoretical speedup: {df['tx_theoretical_speedup'].mean():.1f}x")
    print(f"Median tx theoretical speedup: {df['tx_theoretical_speedup'].median():.1f}x")
    
    print(f"\n--- Dependency Chain Analysis ---")
    print(f"Average dependency chains per block: {df['dependency_chains'].mean():.1f}")
    print(f"Average chain length: {df['avg_chain_length'].mean():.1f}")
    print(f"Max chain length observed: {df['max_chain_length'].max()}")
    print(f"Average longest chain gas: {df['max_chain_gas'].mean():,.0f}")
    print(f"Max longest chain gas: {df['max_chain_gas'].max():,}")
    print(f"Average critical path gas: {df['critical_path_gas'].mean():,.0f}")
    
    print(f"\n--- Gas Utilization Analysis ---")
    print(f"Average gas utilization per block: {df['gas_utilization'].mean():.1f}%")
    print(f"Largest single dependency gas: {df['largest_dependency_gas'].max():,}")
    print(f"Average largest dependency gas: {df['largest_dependency_gas'].mean():,.0f}")
    
    print(f"\n--- Performance Analysis ---")
    print(f"Average analysis time per block: {df['analysis_time_seconds'].mean():.1f}s")
    print(f"Average total time per block: {df['total_block_time_seconds'].mean():.1f}s")
    print(f"Analysis efficiency: {(df['analysis_time_seconds'].sum() / df['total_block_time_seconds'].sum())*100:.1f}% of time spent on analysis")


def calculate_summary_statistics(df):
    """Calculate summary statistics for JSON export."""
    return {
        'total_blocks': len(df),
        'total_transactions': int(df['total_transactions'].sum()),
        'total_gas': int(df['total_gas'].sum()),
        'total_dependencies': int(df['state_dependencies'].sum()),
        
        # Gas-based metrics (primary)
        'total_dependent_gas': int(df['dependent_gas'].sum()),
        'total_independent_gas': int(df['independent_gas'].sum()),
        'dependent_gas_percent': float((df['dependent_gas'].sum() / df['total_gas'].sum()) * 100),
        'independent_gas_percent': float((df['independent_gas'].sum() / df['total_gas'].sum()) * 100),
        'avg_gas_parallelization_percent': float(df['gas_parallelization_percent'].mean()),
        'min_gas_parallelization_percent': float(df['gas_parallelization_percent'].min()),
        'max_gas_parallelization_percent': float(df['gas_parallelization_percent'].max()),
        'avg_gas_theoretical_speedup': float(df['gas_theoretical_speedup'].mean()),
        'median_gas_theoretical_speedup': float(df['gas_theoretical_speedup'].median()),
        'max_gas_theoretical_speedup': float(df['gas_theoretical_speedup'].max()),
        
        # Transaction-based metrics (for comparison)
        'avg_tx_parallelization_percent': float(df['tx_parallelization_percent'].mean()),
        'avg_tx_theoretical_speedup': float(df['tx_theoretical_speedup'].mean()),
        'median_tx_theoretical_speedup': float(df['tx_theoretical_speedup'].median()),
        'max_tx_theoretical_speedup': float(df['tx_theoretical_speedup'].max()),
        
        # Other metrics
        'dependency_rate_percent': float((df['state_dependencies'].sum() / df['total_transactions'].sum()) * 100),
        'avg_gas_utilization_percent': float(df['gas_utilization'].mean()),
        'avg_analysis_time_seconds': float(df['analysis_time_seconds'].mean()),
        'total_analysis_time_minutes': float(df['analysis_time_seconds'].sum() / 60),
        'avg_critical_path_gas': float(df['critical_path_gas'].mean()),
        'avg_longest_chain_gas': float(df['max_chain_gas'].mean())
    }


def generate_visualizations(df, start_block, end_block):
    """Generate comprehensive visualizations."""
    # Create output directory
    output_dir = Path("data/graphs")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Gas-Based Parallelization Potential Over Time
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df['block_number'],
        y=df['gas_parallelization_percent'],
        mode='lines+markers',
        name='Gas Parallelization %',
        line=dict(color='blue', width=2),
        marker=dict(size=6),
        text=[f"Block {b}<br>Gas Parallel: {p:.1f}%<br>Dependencies: {d}<br>Dependent Gas: {dg:,}" 
              for b, p, d, dg in zip(df['block_number'], df['gas_parallelization_percent'], 
                                   df['state_dependencies'], df['dependent_gas'])],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Add transaction-based line for comparison
    fig1.add_trace(go.Scatter(
        x=df['block_number'],
        y=df['tx_parallelization_percent'],
        mode='lines+markers',
        name='TX Parallelization %',
        line=dict(color='lightblue', width=1, dash='dash'),
        marker=dict(size=4),
        text=[f"Block {b}<br>TX Parallel: {p:.1f}%" 
              for b, p in zip(df['block_number'], df['tx_parallelization_percent'])],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig1.update_layout(
        title='Ethereum Parallelization Potential Over Time (Gas-Based vs Transaction-Based)',
        xaxis_title='Block Number',
        yaxis_title='Parallelization Potential (%)',
        template='plotly_white',
        height=500
    )
    fig1.write_html(output_dir / f"parallelization_over_time_{start_block}_{end_block}.html")
    
    # 2. Gas-Based Theoretical Speedup Distribution
    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Gas-Based Speedup Distribution', 'Transaction-Based Speedup Distribution')
    )
    
    fig2.add_trace(go.Histogram(
        x=df['gas_theoretical_speedup'],
        nbinsx=15,
        name='Gas Speedup',
        marker_color='green',
        opacity=0.7
    ), row=1, col=1)
    
    fig2.add_trace(go.Histogram(
        x=df['tx_theoretical_speedup'],
        nbinsx=15,
        name='TX Speedup',
        marker_color='lightgreen',
        opacity=0.7
    ), row=1, col=2)
    
    fig2.update_layout(
        title='Distribution of Theoretical Speedup (Gas vs Transaction Based)',
        template='plotly_white',
        height=500
    )
    fig2.write_html(output_dir / f"speedup_distribution_{start_block}_{end_block}.html")
    
    # 3. Dependencies vs Block Size (Enhanced with Gas Metrics)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df['total_transactions'],
        y=df['dependent_gas'],
        mode='markers',
        marker=dict(
            size=df['gas_theoretical_speedup'],
            sizemode='diameter',
            sizeref=2.*max(df['gas_theoretical_speedup'])/(40.**2),
            sizemin=4,
            color=df['gas_parallelization_percent'],
            colorscale='Viridis',
            colorbar=dict(title="Gas Parallelization %"),
            line=dict(width=1, color='white')
        ),
        text=[f"Block {b}<br>TXs: {t}<br>Deps: {d}<br>Dependent Gas: {dg:,}<br>Gas Speedup: {gs:.1f}x<br>Gas Parallel: {gp:.1f}%<br>Critical Path: {cp:,}" 
              for b, t, d, dg, gs, gp, cp in zip(df['block_number'], df['total_transactions'], 
                                     df['state_dependencies'], df['dependent_gas'],
                                     df['gas_theoretical_speedup'], df['gas_parallelization_percent'],
                                     df['critical_path_gas'])],
        hovertemplate='%{text}<extra></extra>'
    ))
    fig3.update_layout(
        title='Dependent Gas vs Block Size (bubble size = gas speedup, color = gas parallelization %)',
        xaxis_title='Total Transactions in Block',
        yaxis_title='Dependent Gas',
        template='plotly_white',
        height=600
    )
    fig3.write_html(output_dir / f"dependencies_vs_blocksize_{start_block}_{end_block}.html")
    
    # 4. Gas Analysis (Enhanced)
    fig4 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Gas Utilization Over Time', 'Dependent vs Independent Gas', 
                       'Largest Dependency Gas', 'Gas Utilization Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Gas utilization over time
    fig4.add_trace(go.Scatter(
        x=df['block_number'], y=df['gas_utilization'],
        mode='lines+markers', name='Gas Utilization %',
        line=dict(color='red')
    ), row=1, col=1)
    
    # Dependent vs Independent gas (stacked bar)
    fig4.add_trace(go.Bar(
        x=df['block_number'], y=df['dependent_gas'],
        name='Dependent Gas', marker_color='red', opacity=0.7
    ), row=1, col=2)
    fig4.add_trace(go.Bar(
        x=df['block_number'], y=df['independent_gas'],
        name='Independent Gas', marker_color='green', opacity=0.7
    ), row=1, col=2)
    
    # Largest dependency gas
    fig4.add_trace(go.Scatter(
        x=df['block_number'], y=df['largest_dependency_gas'],
        mode='markers', name='Largest Dep Gas',
        marker=dict(size=8, color='orange')
    ), row=2, col=1)
    
    # Gas utilization distribution
    fig4.add_trace(go.Histogram(
        x=df['gas_utilization'], name='Gas Util Distribution',
        marker_color='purple', opacity=0.7
    ), row=2, col=2)
    
    fig4.update_layout(
        title='Gas Analysis Across Blocks',
        template='plotly_white',
        height=800,
        showlegend=True
    )
    fig4.write_html(output_dir / f"gas_analysis_{start_block}_{end_block}.html")
    
    # 5. Dependency Chain Analysis (Gas-Based)
    fig5 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Dependency Chains per Block', 'Max Chain Gas Over Time',
                       'Average Chain Gas vs Length', 'Chain Gas Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Number of dependency chains per block
    fig5.add_trace(go.Scatter(
        x=df['block_number'], y=df['dependency_chains'],
        mode='lines+markers', name='Dependency Chains',
        line=dict(color='purple', width=2),
        text=[f"Block {b}<br>Chains: {c}<br>Total Chain Gas: {tcg:,}" 
              for b, c, tcg in zip(df['block_number'], df['dependency_chains'], df['total_chain_gas'])],
        hovertemplate='%{text}<extra></extra>'
    ), row=1, col=1)
    
    # Max chain gas over time
    fig5.add_trace(go.Scatter(
        x=df['block_number'], y=df['max_chain_gas'],
        mode='lines+markers', name='Max Chain Gas',
        line=dict(color='red', width=2),
        text=[f"Block {b}<br>Max Chain Gas: {mcg:,}<br>Max Chain Length: {mcl}" 
              for b, mcg, mcl in zip(df['block_number'], df['max_chain_gas'], df['max_chain_length'])],
        hovertemplate='%{text}<extra></extra>'
    ), row=1, col=2)
    
    # Average chain gas vs average chain length
    fig5.add_trace(go.Scatter(
        x=df['avg_chain_length'], y=df['avg_chain_gas'],
        mode='markers', name='Gas vs Length',
        marker=dict(size=df['dependency_chains'], sizemode='diameter', 
                   sizeref=2.*max(df['dependency_chains'])/(30.**2), sizemin=4,
                   color='orange'),
        text=[f"Block {b}<br>Avg Chain Length: {acl:.1f}<br>Avg Chain Gas: {acg:,.0f}<br>Chains: {c}" 
              for b, acl, acg, c in zip(df['block_number'], df['avg_chain_length'], 
                                       df['avg_chain_gas'], df['dependency_chains'])],
        hovertemplate='%{text}<extra></extra>'
    ), row=2, col=1)
    
    # Chain gas distribution (histogram of all chain gas amounts)
    all_chain_gas = []
    for chain_gas_list in df['chain_gas_amounts']:
        if chain_gas_list:  # Check if not empty
            all_chain_gas.extend(chain_gas_list)
    
    if all_chain_gas:
        fig5.add_trace(go.Histogram(
            x=all_chain_gas, name='Chain Gas Distribution',
            marker_color='blue', opacity=0.7,
            nbinsx=15
        ), row=2, col=2)
    
    fig5.update_layout(
        title='Dependency Chain Analysis (Gas-Based Metrics)',
        template='plotly_white',
        height=800
    )
    
    # Update axis labels
    fig5.update_xaxes(title_text="Block Number", row=1, col=1)
    fig5.update_yaxes(title_text="Number of Chains", row=1, col=1)
    fig5.update_xaxes(title_text="Block Number", row=1, col=2)
    fig5.update_yaxes(title_text="Max Chain Gas", row=1, col=2)
    fig5.update_xaxes(title_text="Average Chain Length (TXs)", row=2, col=1)
    fig5.update_yaxes(title_text="Average Chain Gas", row=2, col=1)
    fig5.update_xaxes(title_text="Chain Gas Amount", row=2, col=2)
    fig5.update_yaxes(title_text="Frequency", row=2, col=2)
    
    fig5.write_html(output_dir / f"dependency_chains_{start_block}_{end_block}.html")
    
    # 6. Performance Analysis
    fig6 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Analysis Time per Block', 'Dependencies vs Analysis Time',
                       'Transactions vs Analysis Time', 'Performance Over Time')
    )
    
    fig6.add_trace(go.Bar(
        x=df['block_number'], y=df['analysis_time_seconds'],
        name='Analysis Time', marker_color='brown'
    ), row=1, col=1)
    
    fig6.add_trace(go.Scatter(
        x=df['state_dependencies'], y=df['analysis_time_seconds'],
        mode='markers', name='Deps vs Time',
        marker=dict(color='red', size=8)
    ), row=1, col=2)
    
    fig6.add_trace(go.Scatter(
        x=df['total_transactions'], y=df['analysis_time_seconds'],
        mode='markers', name='TXs vs Time',
        marker=dict(color='blue', size=8)
    ), row=2, col=1)
    
    fig6.add_trace(go.Scatter(
        x=df['block_number'], y=df['total_block_time_seconds'],
        mode='lines+markers', name='Total Block Time',
        line=dict(color='green', width=2)
    ), row=2, col=2)
    
    fig6.update_layout(
        title='Performance Analysis',
        template='plotly_white',
        height=800
    )
    fig6.write_html(output_dir / f"performance_analysis_{start_block}_{end_block}.html")
    
    # 7. Summary Dashboard (Enhanced)
    fig7 = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Gas Parallelization % Over Time', 'Gas vs TX Speedup Over Time',
                       'Dependencies Over Time', 'Dependent Gas vs Total Gas',
                       'Gas Utilization', 'Analysis Performance'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Row 1
    fig7.add_trace(go.Scatter(x=df['block_number'], y=df['gas_parallelization_percent'],
                             mode='lines+markers', name='Gas Parallel %', line=dict(color='blue')), row=1, col=1)
    fig7.add_trace(go.Scatter(x=df['block_number'], y=df['gas_theoretical_speedup'],
                             mode='lines+markers', name='Gas Speedup', line=dict(color='green')), row=1, col=2)
    fig7.add_trace(go.Scatter(x=df['block_number'], y=df['tx_theoretical_speedup'],
                             mode='lines+markers', name='TX Speedup', line=dict(color='lightgreen')), row=1, col=2)
    
    # Row 2
    fig7.add_trace(go.Scatter(x=df['block_number'], y=df['state_dependencies'],
                             mode='lines+markers', name='Dependencies', line=dict(color='red')), row=2, col=1)
    fig7.add_trace(go.Scatter(x=df['total_gas'], y=df['dependent_gas'],
                             mode='markers', name='Dependent vs Total Gas', marker=dict(color='orange')), row=2, col=2)
    
    # Row 3
    fig7.add_trace(go.Scatter(x=df['block_number'], y=df['gas_utilization'],
                             mode='lines+markers', name='Gas %', line=dict(color='purple')), row=3, col=1)
    fig7.add_trace(go.Scatter(x=df['block_number'], y=df['analysis_time_seconds'],
                             mode='markers', name='Analysis Time', marker=dict(color='brown')), row=3, col=2)
    
    fig7.update_layout(
        title=f'Ethereum Parallelization Dashboard - Gas-Based Analysis (Blocks {start_block}-{end_block})',
        template='plotly_white',
        height=1200,
        showlegend=False
    )
    fig7.write_html(output_dir / f"dashboard_{start_block}_{end_block}.html")
    
    print(f"   ✅ Generated 7 comprehensive visualizations")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Ethereum Parallelization Analysis (Optimized)')
    parser.add_argument('--blocks', type=int, default=10, help='Number of blocks to analyze (default: 10)')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers (default: 8)')
    
    args = parser.parse_args()
    
    analyze_ethereum_parallelization(args.blocks, args.workers) 