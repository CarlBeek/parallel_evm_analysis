"""
Dependency graph visualization for transaction dependencies.
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
from analysis.state_dependency_analyzer import StateDependency


class DependencyGraphVisualizer:
    """
    Creates and visualizes dependency graphs for transaction analysis.
    """
    
    def __init__(self, database: BlockchainDatabase, output_dir: str = "./data/graphs"):
        """
        Initialize the dependency graph visualizer.
        
        Args:
            database: BlockchainDatabase instance
            output_dir: Directory to save graph outputs
        """
        self.database = database
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def create_dependency_graph(self, block_number: int) -> nx.DiGraph:
        """
        Create a NetworkX directed graph from block dependencies.
        
        Args:
            block_number: Block number to create graph for
            
        Returns:
            NetworkX directed graph
        """
        # Get dependencies from database
        dependencies = self.database.get_dependencies_for_block(block_number)
        transactions = self.database.get_transactions_by_block(block_number)
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add transaction nodes
        for tx in transactions:
            G.add_node(
                tx['hash'],
                transaction_index=tx['transaction_index'],
                from_address=tx['from_address'],
                to_address=tx['to_address'],
                gas_used=tx['gas_used'] or 0,
                status=tx['status'],
                value=int(tx['value']) if tx['value'] else 0
            )
        
        # Add dependency edges
        for dep in dependencies:
            if dep['dependent_tx_hash'] in G.nodes and dep['dependency_tx_hash'] in G.nodes:
                G.add_edge(
                    dep['dependency_tx_hash'],
                    dep['dependent_tx_hash'],
                    dependency_type=dep['dependency_type'],
                    reason=dep['dependency_reason'],
                    gas_impact=dep['gas_impact'],
                    dependent_index=dep['dependent_index'],
                    dependency_index=dep['dependency_index']
                )
        
        self.logger.info(f"Created dependency graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def analyze_graph_structure(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Analyze the structure of the dependency graph.
        
        Args:
            G: NetworkX directed graph
            
        Returns:
            Dictionary with graph analysis results
        """
        analysis = {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_dag': nx.is_directed_acyclic_graph(G),
            'strongly_connected_components': len(list(nx.strongly_connected_components(G))),
            'weakly_connected_components': len(list(nx.weakly_connected_components(G))),
        }
        
        # Calculate centrality measures
        if G.number_of_nodes() > 0:
            analysis['in_degree_centrality'] = nx.in_degree_centrality(G)
            analysis['out_degree_centrality'] = nx.out_degree_centrality(G)
            analysis['betweenness_centrality'] = nx.betweenness_centrality(G)
        
        # Find longest dependency chains
        if nx.is_directed_acyclic_graph(G):
            analysis['longest_path'] = nx.dag_longest_path_length(G)
            analysis['topological_generations'] = len(list(nx.topological_generations(G)))
        else:
            analysis['longest_path'] = 0
            analysis['topological_generations'] = 0
            analysis['cycles'] = list(nx.simple_cycles(G))
        
        # Analyze parallelization potential
        independent_nodes = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0]
        analysis['independent_transactions'] = len(independent_nodes)
        analysis['parallelization_potential'] = len(independent_nodes) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        
        return analysis
    
    def create_interactive_graph(self, block_number: int, max_nodes: int = 100) -> go.Figure:
        """
        Create an interactive Plotly graph visualization.
        
        Args:
            block_number: Block number to visualize
            max_nodes: Maximum number of nodes to display
            
        Returns:
            Plotly figure object
        """
        G = self.create_dependency_graph(block_number)
        
        # Limit graph size for visualization
        if G.number_of_nodes() > max_nodes:
            # Keep nodes with highest degree (most connected)
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:max_nodes]
            G = G.subgraph(top_nodes).copy()
            self.logger.info(f"Limited graph to {max_nodes} most connected nodes")
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Prepare node data
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            text=[f"TX {G.nodes[node]['transaction_index']}" for node in G.nodes()],
            textposition="middle center",
            hovertemplate=(
                "<b>Transaction %{customdata[0]}</b><br>" +
                "Hash: %{customdata[1]}<br>" +
                "From: %{customdata[2]}<br>" +
                "To: %{customdata[3]}<br>" +
                "Gas Used: %{customdata[4]:,}<br>" +
                "Status: %{customdata[5]}<br>" +
                "Value: %{customdata[6]:,} wei<br>" +
                "<extra></extra>"
            ),
            customdata=[
                [
                    G.nodes[node]['transaction_index'],
                    node[:10] + "...",
                    G.nodes[node]['from_address'][:10] + "...",
                    G.nodes[node]['to_address'][:10] + "..." if G.nodes[node]['to_address'] else "None",
                    G.nodes[node]['gas_used'],
                    "Success" if G.nodes[node]['status'] == 1 else "Failed" if G.nodes[node]['status'] == 0 else "Unknown",
                    G.nodes[node]['value']
                ]
                for node in G.nodes()
            ],
            marker=dict(
                size=[max(10, min(30, G.nodes[node]['gas_used'] / 10000)) for node in G.nodes()],
                color=[G.nodes[node]['transaction_index'] for node in G.nodes()],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Transaction Index"),
                line=dict(width=2, color='black')
            ),
            name="Transactions"
        )
        
        # Prepare edge data
        edge_traces = []
        dependency_types = set()
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            dep_type = edge[2]['dependency_type']
            dependency_types.add(dep_type)
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(
                    width=2,
                    color=self._get_dependency_color(dep_type)
                ),
                hovertemplate=(
                    f"<b>{dep_type.title()} Dependency</b><br>" +
                    f"From TX {edge[2]['dependency_index']} → TX {edge[2]['dependent_index']}<br>" +
                    f"Reason: {edge[2]['reason']}<br>" +
                    f"Gas Impact: {edge[2]['gas_impact']:,}<br>" +
                    "<extra></extra>"
                ),
                name=f"{dep_type.title()} Dependencies",
                showlegend=True
            )
            edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(data=[node_trace] + edge_traces)
        
        fig.update_layout(
            title=f"Transaction Dependency Graph - Block {block_number}",
            titlefont_size=16,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text=f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="gray", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    def _get_dependency_color(self, dep_type: str) -> str:
        """Get color for dependency type."""
        colors = {
            'contract': '#FF6B6B',    # Red
            'address': '#4ECDC4',     # Teal
            'event': '#45B7D1',       # Blue
            'function': '#96CEB4'     # Green
        }
        return colors.get(dep_type, '#95A5A6')  # Gray default
    
    def create_dependency_statistics_chart(self, block_number: int) -> go.Figure:
        """
        Create charts showing dependency statistics.
        
        Args:
            block_number: Block number to analyze
            
        Returns:
            Plotly figure with multiple subplots
        """
        dependencies = self.database.get_dependencies_for_block(block_number)
        
        if not dependencies:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(text="No dependencies found", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Prepare data
        df = pd.DataFrame(dependencies)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Dependencies by Type',
                'Gas Impact by Dependency Type',
                'Dependency Confidence Distribution',
                'Transaction Index Dependencies'
            ),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # 1. Dependencies by type (pie chart)
        type_counts = df['dependency_type'].value_counts()
        fig.add_trace(
            go.Pie(labels=type_counts.index, values=type_counts.values, name="Dependency Types"),
            row=1, col=1
        )
        
        # 2. Gas impact by type (bar chart)
        gas_by_type = df.groupby('dependency_type')['gas_impact'].sum()
        fig.add_trace(
            go.Bar(x=gas_by_type.index, y=gas_by_type.values, name="Gas Impact"),
            row=1, col=2
        )
        
        # 3. Confidence distribution (histogram)
        # Note: We don't have confidence in the database, so we'll skip this for now
        fig.add_trace(
            go.Histogram(x=df['gas_impact'], nbinsx=20, name="Gas Impact Distribution"),
            row=2, col=1
        )
        
        # 4. Transaction index dependencies (scatter)
        fig.add_trace(
            go.Scatter(
                x=df['dependency_index'],
                y=df['dependent_index'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['gas_impact'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=df['dependency_type'],
                name="Dependencies"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=f"Dependency Analysis Statistics - Block {block_number}",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def save_graph(self, fig: go.Figure, filename: str, format: str = 'html'):
        """
        Save graph to file.
        
        Args:
            fig: Plotly figure to save
            filename: Output filename
            format: Output format ('html', 'png', 'svg', 'pdf')
        """
        output_path = self.output_dir / f"{filename}.{format}"
        
        if format == 'html':
            fig.write_html(str(output_path))
        elif format == 'png':
            fig.write_image(str(output_path))
        elif format == 'svg':
            fig.write_image(str(output_path))
        elif format == 'pdf':
            fig.write_image(str(output_path))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Saved graph to {output_path}")
    
    def generate_dependency_report(self, block_number: int) -> Dict[str, Any]:
        """
        Generate a comprehensive dependency analysis report.
        
        Args:
            block_number: Block number to analyze
            
        Returns:
            Dictionary with complete analysis results
        """
        # Create dependency graph
        G = self.create_dependency_graph(block_number)
        
        # Analyze graph structure
        graph_analysis = self.analyze_graph_structure(G)
        
        # Get dependency statistics
        dependencies = self.database.get_dependencies_for_block(block_number)
        transactions = self.database.get_transactions_by_block(block_number)
        
        # Calculate additional metrics
        total_gas = sum(tx['gas_used'] or 0 for tx in transactions)
        dependent_gas = sum(dep['gas_impact'] for dep in dependencies)
        
        # Identify critical transactions (high degree centrality)
        critical_transactions = []
        if 'in_degree_centrality' in graph_analysis:
            in_centrality = graph_analysis['in_degree_centrality']
            out_centrality = graph_analysis['out_degree_centrality']
            
            for tx_hash in G.nodes():
                total_centrality = in_centrality.get(tx_hash, 0) + out_centrality.get(tx_hash, 0)
                if total_centrality > 0.1:  # Threshold for "critical"
                    critical_transactions.append({
                        'hash': tx_hash,
                        'transaction_index': G.nodes[tx_hash]['transaction_index'],
                        'in_degree': G.in_degree(tx_hash),
                        'out_degree': G.out_degree(tx_hash),
                        'centrality': total_centrality
                    })
        
        report = {
            'block_number': block_number,
            'total_transactions': len(transactions),
            'total_dependencies': len(dependencies),
            'graph_structure': graph_analysis,
            'gas_analysis': {
                'total_block_gas': total_gas,
                'dependent_gas': dependent_gas,
                'gas_efficiency': (total_gas - dependent_gas) / total_gas if total_gas > 0 else 0
            },
            'dependency_breakdown': {},
            'critical_transactions': sorted(critical_transactions, key=lambda x: x['centrality'], reverse=True)[:10],
            'parallelization_metrics': {
                'independent_transactions': graph_analysis.get('independent_transactions', 0),
                'parallelization_potential': graph_analysis.get('parallelization_potential', 0),
                'longest_dependency_chain': graph_analysis.get('longest_path', 0),
                'dependency_levels': graph_analysis.get('topological_generations', 0)
            }
        }
        
        # Dependency type breakdown
        if dependencies:
            df = pd.DataFrame(dependencies)
            report['dependency_breakdown'] = {
                'by_type': df['dependency_type'].value_counts().to_dict(),
                'gas_by_type': df.groupby('dependency_type')['gas_impact'].sum().to_dict(),
                'avg_gas_by_type': df.groupby('dependency_type')['gas_impact'].mean().to_dict()
            }
        
        return report
    
    def create_clean_dependency_graph(self, block_number: int, use_refined: bool = True) -> go.Figure:
        """
        Create a clean dependency graph visualization similar to the user's example.
        
        This shows:
        - Only transactions involved in dependencies
        - Clear dependency chains (A -> B -> C)
        - Much cleaner layout without independent transaction clutter
        
        Args:
            block_number: Block number to visualize
            use_refined: Whether to use refined dependencies only
            
        Returns:
            Plotly figure object with clean layout
        """
        # Get dependencies and transactions
        if use_refined:
            # Get only refined dependencies (high confidence)
            all_deps = self.database.get_dependencies_for_block(block_number)
            dependencies = [dep for dep in all_deps if dep['dependency_type'].startswith('refined_')]
        else:
            dependencies = self.database.get_dependencies_for_block(block_number)
        
        transactions = self.database.get_transactions_by_block(block_number)
        
        # Create directed graph with ONLY transactions involved in dependencies
        G = nx.DiGraph()
        
        # First, identify which transactions are involved in dependencies
        involved_tx_hashes = set()
        for dep in dependencies:
            involved_tx_hashes.add(dep['dependent_tx_hash'])
            involved_tx_hashes.add(dep['dependency_tx_hash'])
        
        # Add only the involved transaction nodes
        tx_lookup = {tx['hash']: tx for tx in transactions}
        for tx_hash in involved_tx_hashes:
            if tx_hash in tx_lookup:
                tx = tx_lookup[tx_hash]
                G.add_node(
                    tx['hash'],
                    transaction_index=tx['transaction_index'],
                    from_address=tx['from_address'],
                    to_address=tx['to_address'],
                    gas_used=tx['gas_used'] or 0,
                    status=tx['status'],
                    value=int(tx['value']) if tx['value'] else 0
                )
        
        # Add only the refined dependency edges
        for dep in dependencies:
            if dep['dependent_tx_hash'] in G.nodes and dep['dependency_tx_hash'] in G.nodes:
                G.add_edge(
                    dep['dependency_tx_hash'],
                    dep['dependent_tx_hash'],
                    dependency_type=dep['dependency_type'].replace('refined_', ''),
                    reason=dep['dependency_reason'],
                    gas_impact=dep['gas_impact'],
                    dependent_index=dep['dependent_index'],
                    dependency_index=dep['dependency_index']
                )
        
        total_transactions = len(transactions)
        involved_transactions = G.number_of_nodes()
        independent_transactions = total_transactions - len(involved_tx_hashes)
        
        self.logger.info(f"Created clean dependency graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        self.logger.info(f"Showing {involved_transactions} involved transactions out of {total_transactions} total")
        
        # If no dependencies, show a simple message
        if G.number_of_nodes() == 0:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No dependencies found in block {block_number}<br>All {total_transactions} transactions are independent",
                xref="paper", yref="paper", x=0.5, y=0.5, 
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(
                title=f"Clean Transaction Dependency Graph - Block {block_number}",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
            return fig
        
        # Create custom layout for clean visualization
        pos = self._create_hierarchical_layout(G)
        
        # Prepare node data with better styling
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            text=[f"TX{G.nodes[node]['transaction_index']}" for node in G.nodes()],
            textposition="middle center",
            textfont=dict(size=12, color="white", family="Arial Black"),
            hovertemplate=(
                "<b>Transaction %{customdata[0]}</b><br>" +
                "Hash: %{customdata[1]}<br>" +
                "From: %{customdata[2]}<br>" +
                "To: %{customdata[3]}<br>" +
                "Gas Used: %{customdata[4]:,}<br>" +
                "Status: %{customdata[5]}<br>" +
                "Value: %{customdata[6]:,} wei<br>" +
                "<extra></extra>"
            ),
            customdata=[
                [
                    G.nodes[node]['transaction_index'],
                    node[:10] + "...",
                    G.nodes[node]['from_address'][:10] + "...",
                    G.nodes[node]['to_address'][:10] + "..." if G.nodes[node]['to_address'] else "None",
                    G.nodes[node]['gas_used'],
                    "Success" if G.nodes[node]['status'] == 1 else "Failed" if G.nodes[node]['status'] == 0 else "Unknown",
                    G.nodes[node]['value']
                ]
                for node in G.nodes()
            ],
            marker=dict(
                size=25,  # Larger size for better visibility
                color=[self._get_node_color(G, node) for node in G.nodes()],
                line=dict(width=3, color='white')
            ),
            name="Transactions",
            showlegend=False
        )
        
        # Prepare edge data with arrows
        edge_traces = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            dep_type = edge[2]['dependency_type']
            
            # Create arrow line
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(
                    width=4,
                    color=self._get_dependency_color(dep_type)
                ),
                hovertemplate=(
                    f"<b>{dep_type.title()} Dependency</b><br>" +
                    f"TX{edge[2]['dependency_index']} → TX{edge[2]['dependent_index']}<br>" +
                    f"Reason: {edge[2]['reason']}<br>" +
                    f"Gas Impact: {edge[2]['gas_impact']:,}<br>" +
                    "<extra></extra>"
                ),
                name=f"{dep_type.title()} Dependencies",
                showlegend=True
            )
            edge_traces.append(edge_trace)
            
            # Add arrowhead
            arrow_trace = self._create_arrow(x0, y0, x1, y1, dep_type)
            edge_traces.append(arrow_trace)
        
        # Create figure with clean styling
        fig = go.Figure(data=[node_trace] + edge_traces)
        
        fig.update_layout(
            title=f"Clean Transaction Dependency Graph - Block {block_number}",
            titlefont_size=18,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=80,l=40,r=40,t=80),
            annotations=[
                dict(
                    text=f"Showing {involved_transactions} dependent transactions out of {total_transactions} total<br>" +
                         f"{independent_transactions} independent transactions not shown ({independent_transactions/total_transactions*100:.1f}% parallelizable)",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.15,
                    xanchor='center', yanchor='top',
                    font=dict(color="gray", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1200,
            height=600
        )
        
        return fig
    
    def _create_hierarchical_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """
        Create a hierarchical layout for dependency chains.
        
        Args:
            G: NetworkX directed graph
            
        Returns:
            Dictionary mapping node to (x, y) position
        """
        pos = {}
        
        if G.number_of_nodes() == 0:
            return pos
        
        # Find connected components
        weakly_connected = list(nx.weakly_connected_components(G))
        
        y_offset = 0
        x_spacing = 3.0
        y_spacing = 2.0
        
        for i, component in enumerate(weakly_connected):
            subgraph = G.subgraph(component)
            
            if subgraph.number_of_edges() == 0:
                # Single isolated node
                node = list(component)[0]
                pos[node] = (0, y_offset)
            elif nx.is_directed_acyclic_graph(subgraph):
                # Use topological sort for clean dependency chain
                try:
                    topo_order = list(nx.topological_sort(subgraph))
                    for j, node in enumerate(topo_order):
                        pos[node] = (j * x_spacing, y_offset)
                except:
                    # Fallback to simple horizontal layout
                    for j, node in enumerate(component):
                        pos[node] = (j * x_spacing, y_offset)
            else:
                # Handle cycles with circular layout
                cycle_pos = nx.circular_layout(subgraph)
                for node, (x, y) in cycle_pos.items():
                    pos[node] = (x * 2 + i * 6, y * 2 + y_offset)
            
            y_offset -= y_spacing
        
        return pos
    
    def _get_node_color(self, G: nx.DiGraph, node: str) -> str:
        """Get color for node based on its role in the dependency graph."""
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        
        if in_degree == 0 and out_degree == 0:
            return '#95A5A6'  # Gray for independent
        elif in_degree == 0:
            return '#2ECC71'  # Green for source (no dependencies)
        elif out_degree == 0:
            return '#E74C3C'  # Red for sink (nothing depends on it)
        else:
            return '#3498DB'  # Blue for intermediate
    
    def _create_arrow(self, x0: float, y0: float, x1: float, y1: float, dep_type: str) -> go.Scatter:
        """Create an arrow to show dependency direction."""
        # Calculate arrow position (80% along the line)
        arrow_x = x0 + 0.8 * (x1 - x0)
        arrow_y = y0 + 0.8 * (y1 - y0)
        
        return go.Scatter(
            x=[arrow_x],
            y=[arrow_y],
            mode='markers',
            marker=dict(
                symbol='triangle-right',
                size=8,
                color=self._get_dependency_color(dep_type),
                line=dict(width=1, color='white')
            ),
            showlegend=False,
            hoverinfo='skip'
        )
    
    def create_gantt_chart(self, block_number: int, use_refined: bool = True) -> go.Figure:
        """
        Create a Gantt chart visualization of transaction dependencies.
        
        This shows:
        - Transaction execution timeline (X-axis: time/order)
        - Each transaction as a horizontal bar (width proportional to gas used)
        - Dependencies as connections/constraints
        - Parallel execution opportunities clearly visible
        
        Args:
            block_number: Block number to visualize
            use_refined: Whether to use refined dependencies only
            
        Returns:
            Plotly figure object with Gantt chart
        """
        # Get dependencies and transactions
        if use_refined:
            all_deps = self.database.get_dependencies_for_block(block_number)
            dependencies = [dep for dep in all_deps if dep['dependency_type'].startswith('refined_')]
        else:
            dependencies = self.database.get_dependencies_for_block(block_number)
        
        transactions = self.database.get_transactions_by_block(block_number)
        
        # Sort transactions by index
        transactions.sort(key=lambda x: x['transaction_index'])
        
        # Create dependency mapping for simple scheduling
        dependency_map = {}
        reverse_dependency_map = {}  # Track what each transaction depends on
        for dep in dependencies:
            dependent = dep['dependent_tx_hash']
            dependency = dep['dependency_tx_hash']
            if dependent not in dependency_map:
                dependency_map[dependent] = []
            dependency_map[dependent].append(dependency)
            
            if dependency not in reverse_dependency_map:
                reverse_dependency_map[dependency] = []
            reverse_dependency_map[dependency].append(dependent)
        
        # Calculate proper parallel execution schedule
        tx_lookup = {tx['hash']: tx for tx in transactions}
        execution_schedule = {}
        
        # Independent transactions start at time 0
        for tx in transactions:
            tx_hash = tx['hash']
            if tx_hash not in dependency_map and tx_hash not in reverse_dependency_map:
                execution_schedule[tx_hash] = 0
        
        # For dependent transactions, calculate proper ordering
        def calculate_execution_time(tx_hash, visited=None):
            if visited is None:
                visited = set()
            
            if tx_hash in visited:
                return 0  # Avoid cycles
            
            if tx_hash in execution_schedule:
                return execution_schedule[tx_hash]
            
            visited.add(tx_hash)
            
            # If this transaction has dependencies, it must start after all of them finish
            if tx_hash in dependency_map:
                max_dependency_finish_time = 0
                for dep_hash in dependency_map[tx_hash]:
                    dep_start_time = calculate_execution_time(dep_hash, visited.copy())
                    dep_tx = tx_lookup.get(dep_hash)
                    if dep_tx:
                        dep_duration = max(1000, dep_tx['gas_used'] or 0)
                        dep_finish_time = dep_start_time + dep_duration
                        max_dependency_finish_time = max(max_dependency_finish_time, dep_finish_time)
                
                execution_schedule[tx_hash] = max_dependency_finish_time
            else:
                # No dependencies, can start at time 0
                execution_schedule[tx_hash] = 0
            
            return execution_schedule[tx_hash]
        
        # Calculate execution times for all transactions
        for tx in transactions:
            calculate_execution_time(tx['hash'])
        
        # Create hierarchical ordering for better visual layout
        def create_hierarchical_order():
            ordered_transactions = []
            processed = set()
            
            # First, identify all dependency chains and calculate their total gas
            def find_chain_components():
                # Build a graph of dependencies to find connected components
                import networkx as nx
                dep_graph = nx.DiGraph()
                
                # Add all transactions as nodes
                for tx in transactions:
                    dep_graph.add_node(tx['hash'], gas_used=tx['gas_used'] or 0)
                
                # Add dependency edges
                for dep in dependencies:
                    dependent = dep['dependent_tx_hash']
                    dependency = dep['dependency_tx_hash']
                    if dependent in tx_lookup and dependency in tx_lookup:
                        dep_graph.add_edge(dependency, dependent)
                
                # Find weakly connected components (chains)
                components = list(nx.weakly_connected_components(dep_graph))
                
                # Calculate total gas for each component and sort by gas
                component_gas = []
                for component in components:
                    total_gas = sum(tx_lookup[tx_hash]['gas_used'] or 0 for tx_hash in component if tx_hash in tx_lookup)
                    max_gas = max((tx_lookup[tx_hash]['gas_used'] or 0 for tx_hash in component if tx_hash in tx_lookup), default=0)
                    component_gas.append((total_gas, max_gas, component))
                
                # Sort by total gas (descending), then by max gas (descending)
                component_gas.sort(key=lambda x: (x[0], x[1]), reverse=True)
                
                return [comp[2] for comp in component_gas]
            
            # Get dependency chains sorted by gas
            gas_sorted_chains = find_chain_components()
            
            # Helper function to add transaction and its immediate dependents within a chain
            def add_transaction_with_dependents(tx_hash, depth=0, chain_transactions=None):
                if tx_hash in processed or tx_hash not in tx_lookup:
                    return
                
                # Only process if this transaction is part of the current chain
                if chain_transactions is not None and tx_hash not in chain_transactions:
                    return
                
                tx = tx_lookup[tx_hash]
                processed.add(tx_hash)
                
                # Add this transaction
                ordered_transactions.append({
                    'transaction': tx,
                    'depth': depth,
                    'execution_time': execution_schedule.get(tx_hash, 0)
                })
                
                # Immediately add all transactions that depend directly on this one
                if tx_hash in reverse_dependency_map:
                    dependents = reverse_dependency_map[tx_hash]
                    # Sort dependents by their gas usage (descending), then by transaction index
                    dependents.sort(key=lambda dep_hash: (
                        -(tx_lookup[dep_hash]['gas_used'] or 0),
                        tx_lookup[dep_hash]['transaction_index']
                    ))
                    for dependent_hash in dependents:
                        if dependent_hash not in processed and (chain_transactions is None or dependent_hash in chain_transactions):
                            add_transaction_with_dependents(dependent_hash, depth + 1, chain_transactions)
            
            # Process each chain in gas-sorted order
            for chain in gas_sorted_chains:
                chain_tx_hashes = [tx_hash for tx_hash in chain if tx_hash in tx_lookup]
                
                if not chain_tx_hashes:
                    continue
                
                # Find root transactions in this chain (no dependencies within the chain)
                chain_roots = []
                for tx_hash in chain_tx_hashes:
                    if tx_hash not in dependency_map or not any(dep in chain for dep in dependency_map[tx_hash]):
                        chain_roots.append(tx_hash)
                
                # If no clear roots, start with the transaction with highest gas
                if not chain_roots:
                    chain_roots = [max(chain_tx_hashes, key=lambda x: tx_lookup[x]['gas_used'] or 0)]
                
                # Sort roots by gas usage (descending)
                chain_roots.sort(key=lambda tx_hash: -(tx_lookup[tx_hash]['gas_used'] or 0))
                
                # Process each root and its dependents within this chain
                for root_hash in chain_roots:
                    if root_hash not in processed:
                        add_transaction_with_dependents(root_hash, 0, set(chain))
                
                # Add any remaining unprocessed transactions from this chain
                remaining_in_chain = [tx_hash for tx_hash in chain_tx_hashes if tx_hash not in processed]
                remaining_in_chain.sort(key=lambda tx_hash: (
                    -(tx_lookup[tx_hash]['gas_used'] or 0),
                    execution_schedule.get(tx_hash, 0),
                    tx_lookup[tx_hash]['transaction_index']
                ))
                
                for tx_hash in remaining_in_chain:
                    if tx_hash not in processed:
                        add_transaction_with_dependents(tx_hash, 0, set(chain))
            
            # Add any remaining independent transactions, sorted by gas
            remaining_independent = [tx['hash'] for tx in transactions if tx['hash'] not in processed]
            remaining_independent.sort(key=lambda tx_hash: (
                -(tx_lookup[tx_hash]['gas_used'] or 0),
                tx_lookup[tx_hash]['transaction_index']
            ))
            
            for tx_hash in remaining_independent:
                if tx_hash not in processed:
                    tx = tx_lookup[tx_hash]
                    ordered_transactions.append({
                        'transaction': tx,
                        'depth': 0,
                        'execution_time': execution_schedule.get(tx_hash, 0)
                    })
                    processed.add(tx_hash)
            
            return ordered_transactions
        
        # Get hierarchical ordering
        hierarchical_order = create_hierarchical_order()
        
        # Create Gantt chart data with hierarchical ordering
        gantt_data = []
        
        for i, item in enumerate(hierarchical_order):
            tx = item['transaction']
            depth = item['depth']
            tx_hash = tx['hash']
            tx_index = tx['transaction_index']
            gas_used = tx['gas_used'] or 0
            
            # Get execution start time from schedule
            start_time = execution_schedule.get(tx_hash, tx_index)
            # Duration is simply the gas used (no scaling)
            duration = max(1000, gas_used)  # Minimum 1000 for visibility
            
            # Determine color based on dependency status
            if tx_hash in dependency_map:
                color = '#E74C3C'  # Red for dependent transactions
                category = 'Dependent'
            elif tx_hash in reverse_dependency_map:
                color = '#3498DB'  # Blue for transactions that others depend on
                category = 'Dependency'
            else:
                color = '#2ECC71'  # Green for independent transactions
                category = 'Independent'
            
            # Create task name with indentation to show hierarchy
            indent = "  " * depth  # Two spaces per depth level
            task_name = f'{indent}TX{tx_index}'
            
            gantt_data.append({
                'Task': task_name,
                'Start': start_time,
                'Finish': start_time + duration,
                'Duration': duration,
                'Resource': category,
                'Description': f'TX{tx_index} - {tx["from_address"][:10]}... → {tx["to_address"][:10] if tx["to_address"] else "Contract Creation"}...',
                'Gas': gas_used,
                'Status': 'Success' if tx['status'] == 1 else 'Failed',
                'Hash': tx_hash,
                'Color': color,
                'TxIndex': tx_index,
                'Depth': depth,
                'HierarchicalIndex': i  # For Y-axis positioning
            })
        
        # Create the Gantt chart
        fig = go.Figure()
        
        # Sort gantt_data by hierarchical index for proper Y-axis ordering
        gantt_data.sort(key=lambda x: x['HierarchicalIndex'])
        
        # Group by category for legend
        categories = {}
        for data in gantt_data:
            category = data['Resource']
            if category not in categories:
                categories[category] = []
            categories[category].append(data)
        
        # Add bars for each category
        for category, items in categories.items():
            fig.add_trace(go.Bar(
                name=category,
                x=[item['Duration'] for item in items],
                y=[item['Task'] for item in items],
                base=[item['Start'] for item in items],
                orientation='h',
                marker=dict(color=items[0]['Color']),
                text=[f"TX{item['TxIndex']}" for item in items],
                textposition='inside',
                hovertemplate=(
                    "<b>%{y}</b><br>" +
                    "Start: %{base}<br>" +
                    "Gas Used: %{x:,}<br>" +
                    "Status: %{customdata[0]}<br>" +
                    "Hash: %{customdata[1]}<br>" +
                    "Depth: %{customdata[2]}<br>" +
                    "<extra></extra>"
                ),
                customdata=[[item['Status'], item['Hash'][:10] + "...", item['Depth']] for item in items],
                showlegend=True
            ))
        
        # Note: Removed dependency arrows for cleaner visualization
        # Dependencies are now shown through adjacent positioning and color coding
        
        # Calculate simple metrics
        total_transactions = len(transactions)
        involved_tx_hashes = set()
        for dep in dependencies:
            involved_tx_hashes.add(dep['dependent_tx_hash'])
            involved_tx_hashes.add(dep['dependency_tx_hash'])
        
        independent_txs = total_transactions - len(involved_tx_hashes)
        parallelization_potential = (independent_txs / total_transactions * 100) if total_transactions > 0 else 0
        
        fig.update_layout(
            title=f'Transaction Dependency Gantt Chart - Block {block_number}',
            xaxis_title='Gas Used (Duration)',
            yaxis_title='Transactions',
            barmode='overlay',
            height=max(300, len(transactions) * 6),  # Further reduced: min 300px, 6px per transaction
            width=1200,
            showlegend=True,
            margin=dict(l=100, r=50, t=80, b=50),  # Tighter margins
            bargap=0.1,  # Reduce gap between bars for more compact layout
            yaxis=dict(
                categoryorder="array",
                categoryarray=[item['Task'] for item in reversed(gantt_data)],  # Reverse for top-down
                tickfont=dict(size=10),  # Smaller font
                showgrid=False,  # Remove grid lines for cleaner look
                zeroline=False,
                tickmode='linear',
                dtick=1  # Show every transaction tick but more compact
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1,
                tickfont=dict(size=11)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add analysis annotation
        fig.add_annotation(
            text=f"Dependency Analysis:<br>" +
                 f"Total Transactions: {total_transactions}<br>" +
                 f"Dependencies: {len(dependencies)}<br>" +
                 f"Independent: {independent_txs} ({parallelization_potential:.1f}%)<br>" +
                 f"Bar width = Gas Used",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor='left', yanchor='top',
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=10)
        )
        
        return fig 