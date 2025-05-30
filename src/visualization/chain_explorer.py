"""
Enhanced Chain Explorer Visualization
Integrates with existing dependency graph visualizer to provide chain-specific analysis.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import json
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.database import BlockchainDatabase
from visualization.dependency_graph import DependencyGraphVisualizer


class ChainExplorer:
    """
    Enhanced chain visualization that integrates with existing dependency graph visualizer.
    Provides chain-specific analysis and interactive exploration capabilities.
    """
    
    def __init__(self, database: BlockchainDatabase, output_dir: str = "./data/graphs"):
        """
        Initialize the chain explorer.
        
        Args:
            database: BlockchainDatabase instance
            output_dir: Directory to save graph outputs
        """
        self.database = database
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Use existing visualizer for base functionality
        self.base_visualizer = DependencyGraphVisualizer(database, output_dir)
    
    def create_interactive_chain_explorer(self, block_number: int) -> go.Figure:
        """
        Create an interactive chain explorer that allows drilling down into specific chains.
        
        Args:
            block_number: Block number to explore
            
        Returns:
            Interactive Plotly figure with chain exploration capabilities
        """
        # Get chain data using our analysis functions
        from analyze_dependencies import get_block_dependency_chains
        chain_data = get_block_dependency_chains(self.database, block_number)
        
        if not chain_data or not chain_data['chains']:
            fig = go.Figure()
            fig.add_annotation(text=f"No dependency chains found in block {block_number}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create subplots for multi-perspective view
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Chain Network Overview', 'Chain Length Distribution',
                'Chain Timeline', 'Chain Gas Analysis'
            ],
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        chains = chain_data['chains']
        
        # 1. Chain Network Overview (top-left)
        # Create a network layout showing chains as connected components
        for i, chain in enumerate(chains[:10]):  # Limit to top 10 chains for clarity
            x_positions = list(range(len(chain['transactions'])))
            y_position = [i] * len(chain['transactions'])
            
            # Add chain as a line
            fig.add_trace(
                go.Scatter(
                    x=x_positions,
                    y=y_position,
                    mode='lines+markers',
                    name=f"Chain {i+1} ({chain['length']} txs)",
                    hovertemplate=(
                        f"<b>Chain {i+1}</b><br>" +
                        "Length: %{customdata[0]} transactions<br>" +
                        "Total Gas: %{customdata[1]:,}<br>" +
                        "Source Tx: %{customdata[2]}<br>" +
                        "Sink Tx: %{customdata[3]}<br>" +
                        "<extra></extra>"
                    ),
                    customdata=[[
                        chain['length'],
                        chain['total_gas'],
                        chain['tx_details'][0]['index'],
                        chain['tx_details'][-1]['index']
                    ]] * len(chain['transactions']),
                    line=dict(width=3),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
        
        # 2. Chain Length Distribution (top-right)
        length_dist = {}
        for chain in chains:
            length = chain['length']
            length_dist[length] = length_dist.get(length, 0) + 1
        
        lengths = list(length_dist.keys())
        counts = list(length_dist.values())
        
        fig.add_trace(
            go.Bar(
                x=lengths,
                y=counts,
                name='Chain Count',
                marker=dict(color='skyblue'),
                hovertemplate=(
                    "Length: %{x} transactions<br>" +
                    "Count: %{y} chains<br>" +
                    "<extra></extra>"
                )
            ),
            row=1, col=2
        )
        
        # 3. Chain Timeline (bottom-left)
        # Show transaction indices for each chain
        for i, chain in enumerate(chains[:5]):  # Top 5 longest chains
            tx_indices = [detail['index'] for detail in chain['tx_details']]
            
            fig.add_trace(
                go.Scatter(
                    x=tx_indices,
                    y=[i] * len(tx_indices),
                    mode='markers+lines',
                    name=f"Chain {i+1} Timeline",
                    marker=dict(size=10),
                    hovertemplate=(
                        f"<b>Chain {i+1} - Step %{{customdata[0]}}</b><br>" +
                        "Transaction Index: %{x}<br>" +
                        "Gas Used: %{customdata[1]:,}<br>" +
                        "Hash: %{customdata[2]}<br>" +
                        "<extra></extra>"
                    ),
                    customdata=[
                        [j+1, detail['gas'], chain['transactions'][j][:16] + "..."]
                        for j, detail in enumerate(chain['tx_details'])
                    ]
                ),
                row=2, col=1
            )
        
        # 4. Chain Gas Analysis (bottom-right)
        chain_lengths = [chain['length'] for chain in chains]
        total_gas = [chain['total_gas'] for chain in chains]
        avg_gas = [gas / length for gas, length in zip(total_gas, chain_lengths)]
        
        fig.add_trace(
            go.Scatter(
                x=chain_lengths,
                y=avg_gas,
                mode='markers',
                name='Gas per Transaction',
                marker=dict(
                    size=[gas / 50000 for gas in total_gas],  # Size by total gas
                    color=total_gas,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Total Gas")
                ),
                hovertemplate=(
                    "Chain Length: %{x} transactions<br>" +
                    "Avg Gas/Tx: %{y:,.0f}<br>" +
                    "Total Gas: %{customdata:,}<br>" +
                    "<extra></extra>"
                ),
                customdata=total_gas
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Chain Explorer - Block {block_number}",
            showlegend=False
        )
        
        # Update axes titles
        fig.update_xaxes(title_text="Transaction Position in Chain", row=1, col=1)
        fig.update_yaxes(title_text="Chain ID", row=1, col=1)
        
        fig.update_xaxes(title_text="Chain Length", row=1, col=2)
        fig.update_yaxes(title_text="Number of Chains", row=1, col=2)
        
        fig.update_xaxes(title_text="Transaction Index in Block", row=2, col=1)
        fig.update_yaxes(title_text="Chain ID", row=2, col=1)
        
        fig.update_xaxes(title_text="Chain Length", row=2, col=2)
        fig.update_yaxes(title_text="Mean Gas per Transaction", row=2, col=2)
        
        return fig
    
    def create_chain_topology_view(self, block_number: int) -> go.Figure:
        """
        Create a detailed topology view showing chain structure and intersections.
        
        Args:
            block_number: Block number to analyze
            
        Returns:
            Plotly figure showing chain topology
        """
        # Get chain data
        from analyze_dependencies import get_block_dependency_chains
        chain_data = get_block_dependency_chains(self.database, block_number)
        
        if not chain_data or not chain_data['chains']:
            fig = go.Figure()
            fig.add_annotation(text=f"No chains found in block {block_number}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Use the NetworkX graph from chain_data
        G = chain_data['graph']
        chains = chain_data['chains']
        
        # Create layout for the full graph
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Color code nodes by which chain they belong to
        node_colors = {}
        node_chains = {}
        
        for i, chain in enumerate(chains):
            color = px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
            for tx_hash in chain['transactions']:
                node_colors[tx_hash] = color
                node_chains[tx_hash] = i
        
        # Create traces for each chain
        fig = go.Figure()
        
        # Add edges first (so they appear behind nodes)
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{edge[0][:8]}... → {edge[1][:8]}...")
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines',
            name='Dependencies'
        ))
        
        # Add nodes for each chain
        for i, chain in enumerate(chains):
            chain_nodes = chain['transactions']
            if not chain_nodes:
                continue
                
            node_x = [pos[node][0] for node in chain_nodes if node in pos]
            node_y = [pos[node][1] for node in chain_nodes if node in pos]
            
            if not node_x:
                continue
            
            color = px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
            
            # Get transaction details for hover
            hover_info = []
            for tx_hash in chain_nodes:
                if tx_hash in chain_data['dependency_details']:
                    detail = chain_data['dependency_details'][tx_hash]
                    hover_info.append([
                        detail['index'],
                        tx_hash[:16] + "...",
                        detail['gas'],
                        detail['from'][:10] + "..." if detail['from'] else "N/A",
                        detail['to'][:10] + "..." if detail['to'] else "N/A"
                    ])
                else:
                    hover_info.append([0, tx_hash[:16] + "...", 0, "N/A", "N/A"])
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                marker=dict(
                    size=15,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                name=f'Chain {i+1} ({len(chain_nodes)} txs)',
                hovertemplate=(
                    f"<b>Chain {i+1}</b><br>" +
                    "Tx Index: %{customdata[0]}<br>" +
                    "Hash: %{customdata[1]}<br>" +
                    "Gas: %{customdata[2]:,}<br>" +
                    "From: %{customdata[3]}<br>" +
                    "To: %{customdata[4]}<br>" +
                    "<extra></extra>"
                ),
                customdata=hover_info
            ))
        
        fig.update_layout(
            title=f"Chain Topology - Block {block_number}",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text=f"Block {block_number}: {len(chains)} chains, {len(G.nodes())} transactions",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='#888', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    def create_contract_chain_analysis(self, block_number: int) -> go.Figure:
        """
        Create analysis showing which contracts are involved in the longest chains.
        
        Args:
            block_number: Block number to analyze
            
        Returns:
            Plotly figure showing contract involvement in chains
        """
        # Get chain data
        from analyze_dependencies import get_block_dependency_chains
        chain_data = get_block_dependency_chains(self.database, block_number)
        
        if not chain_data or not chain_data['chains']:
            fig = go.Figure()
            fig.add_annotation(text=f"No chains found in block {block_number}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Analyze contract involvement
        contract_chains = {}
        contract_gas = {}
        
        for i, chain in enumerate(chain_data['chains']):
            for tx_detail in chain['tx_details']:
                if tx_detail['to']:
                    contract = tx_detail['to']
                    if contract not in contract_chains:
                        contract_chains[contract] = []
                        contract_gas[contract] = 0
                    
                    contract_chains[contract].append({
                        'chain_id': i,
                        'chain_length': chain['length'],
                        'gas': tx_detail['gas']
                    })
                    contract_gas[contract] += tx_detail['gas']
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Contract Chain Involvement', 'Gas Usage by Contract',
                'Chain Length by Contract', 'Transaction Count by Contract'
            ]
        )
        
        # Sort contracts by total gas usage
        sorted_contracts = sorted(contract_gas.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 1. Contract chain involvement
        contract_names = []
        chain_counts = []
        max_chain_lengths = []
        total_gas_values = []
        
        for contract, total_gas in sorted_contracts:
            # Truncate contract address for display
            short_name = contract[:8] + "..." if len(contract) > 8 else contract
            contract_names.append(short_name)
            
            chains = contract_chains[contract]
            chain_counts.append(len(set(c['chain_id'] for c in chains)))
            max_chain_lengths.append(max(c['chain_length'] for c in chains))
            total_gas_values.append(total_gas)
        
        fig.add_trace(
            go.Bar(
                x=contract_names,
                y=chain_counts,
                name='Chain Count',
                marker=dict(color='lightblue'),
                hovertemplate=(
                    "Contract: %{x}<br>" +
                    "Chains Involved: %{y}<br>" +
                    "<extra></extra>"
                )
            ),
            row=1, col=1
        )
        
        # 2. Gas usage by contract
        fig.add_trace(
            go.Bar(
                x=contract_names,
                y=total_gas_values,
                name='Total Gas',
                marker=dict(color='orange'),
                hovertemplate=(
                    "Contract: %{x}<br>" +
                    "Total Gas: %{y:,}<br>" +
                    "<extra></extra>"
                )
            ),
            row=1, col=2
        )
        
        # 3. Max chain length by contract
        fig.add_trace(
            go.Bar(
                x=contract_names,
                y=max_chain_lengths,
                name='Max Chain Length',
                marker=dict(color='green'),
                hovertemplate=(
                    "Contract: %{x}<br>" +
                    "Max Chain Length: %{y}<br>" +
                    "<extra></extra>"
                )
            ),
            row=2, col=1
        )
        
        # 4. Transaction count by contract
        tx_counts = [len(contract_chains[contract]) for contract, _ in sorted_contracts]
        
        fig.add_trace(
            go.Bar(
                x=contract_names,
                y=tx_counts,
                name='Transaction Count',
                marker=dict(color='purple'),
                hovertemplate=(
                    "Contract: %{x}<br>" +
                    "Transactions: %{y}<br>" +
                    "<extra></extra>"
                )
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text=f"Contract Chain Analysis - Block {block_number}",
            showlegend=False
        )
        
        return fig
    
    def save_chain_analysis(self, block_number: int, filename_prefix: str = "chain_analysis") -> Dict[str, str]:
        """
        Generate and save all chain analysis visualizations for a block.
        
        Args:
            block_number: Block number to analyze
            filename_prefix: Prefix for saved files
            
        Returns:
            Dictionary of saved file paths
        """
        saved_files = {}
        
        try:
            # 1. Interactive Chain Explorer
            explorer_fig = self.create_interactive_chain_explorer(block_number)
            explorer_path = self.output_dir / f"{filename_prefix}_explorer_block_{block_number}.html"
            explorer_fig.write_html(str(explorer_path))
            saved_files['chain_explorer'] = str(explorer_path)
            self.logger.info(f"Saved chain explorer to {explorer_path}")
            
            # 2. Chain Topology View
            topology_fig = self.create_chain_topology_view(block_number)
            topology_path = self.output_dir / f"{filename_prefix}_topology_block_{block_number}.html"
            topology_fig.write_html(str(topology_path))
            saved_files['chain_topology'] = str(topology_path)
            self.logger.info(f"Saved chain topology to {topology_path}")
            
            # 3. Contract Chain Analysis
            contract_fig = self.create_contract_chain_analysis(block_number)
            contract_path = self.output_dir / f"{filename_prefix}_contracts_block_{block_number}.html"
            contract_fig.write_html(str(contract_path))
            saved_files['contract_analysis'] = str(contract_path)
            self.logger.info(f"Saved contract analysis to {contract_path}")
            
            # 4. Also generate base visualizer outputs for comparison
            base_gantt = self.base_visualizer.create_gantt_chart(block_number, use_refined=True)
            gantt_path = self.output_dir / f"{filename_prefix}_gantt_block_{block_number}.html"
            base_gantt.write_html(str(gantt_path))
            saved_files['gantt_chart'] = str(gantt_path)
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Error generating chain analysis for block {block_number}: {e}")
            return saved_files 