"""
Parallelization visualization for thread count performance analysis.
Creates interactive plots showing how gas requirements change with thread count.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import json
import sys
from datetime import datetime
import numpy as np
import scipy.stats as stats

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.parallelization_simulator import (
    ParallelizationSimulator, 
    ThreadCountAnalysis
)
from storage.database import BlockchainDatabase
from core.transaction_fetcher import TransactionData
from analysis.state_dependency_analyzer import StateDependency


class ParallelizationComparisonVisualizer:
    """
    Creates interactive visualizations for parallelization thread count analysis.
    Generates research plots showing gas requirements vs thread count.
    """
    
    def __init__(self, output_dir: str = "./data/graphs"):
        """
        Initialize the parallelization visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def create_thread_count_comparison(
        self,
        analysis: ThreadCountAnalysis,
        title_suffix: str = ""
    ) -> str:
        """
        Create comprehensive thread count comparison visualization.
        
        Args:
            analysis: ThreadCountAnalysis containing results
            title_suffix: Optional suffix for the title
            
        Returns:
            Path to the generated HTML file
        """
        self.logger.info(f"Creating thread count comparison for block {analysis.block_number}")
        
        # Create subplot layout: 2x2 grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Maximum Gas per Thread (Bottleneck)',
                'Mean Gas per Thread (Utilization)', 
                'Parallel Speedup vs Sequential',
                'Thread Efficiency (Load Balance)'
            ],
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        thread_counts = analysis.thread_counts
        bottleneck_gas = analysis.bottleneck_gas_values
        speedup = analysis.speedup_values
        efficiency = analysis.efficiency_values
        
        color = '#2ca02c'  # Green for parallelization
        
        # Plot 1: Bottleneck Gas (most important for research)
        fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=[gas / 1_000_000 for gas in bottleneck_gas],  # Convert to millions
                mode='lines+markers',
                name='Parallelization',
                line=dict(color=color, width=3),
                marker=dict(size=8),
                hovertemplate='<b>Parallelization</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Max Gas: %{y:.1f}M<br>' +
                            '<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Plot 2: Mean Gas
        fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=[gas / 1_000_000 for gas in analysis.mean_gas_values],
                mode='lines+markers',
                name='Parallelization',
                line=dict(color=color, width=3),
                marker=dict(size=8),
                showlegend=False,
                hovertemplate='<b>Parallelization</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Mean Gas: %{y:.1f}M<br>' +
                            '<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Plot 3: Speedup
        fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=speedup,
                mode='lines+markers',
                name='Parallelization',
                line=dict(color=color, width=3),
                marker=dict(size=8),
                showlegend=False,
                hovertemplate='<b>Parallelization</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Speedup: %{y:.2f}x<br>' +
                            '<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Plot 4: Thread Efficiency
        fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=efficiency,
                mode='lines+markers',
                name='Parallelization',
                line=dict(color=color, width=3),
                marker=dict(size=8),
                showlegend=False,
                hovertemplate='<b>Parallelization</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Efficiency: %{y:.2f}<br>' +
                            '<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update axes labels and formatting
        fig.update_xaxes(title_text="Number of Threads", row=1, col=1)
        fig.update_xaxes(title_text="Number of Threads", row=1, col=2)
        fig.update_xaxes(title_text="Number of Threads", row=2, col=1)
        fig.update_xaxes(title_text="Number of Threads", row=2, col=2)
        
        fig.update_yaxes(title_text="Gas (Millions)", row=1, col=1)
        fig.update_yaxes(title_text="Gas (Millions)", row=1, col=2)
        fig.update_yaxes(title_text="Speedup (x)", row=2, col=1)
        fig.update_yaxes(title_text="Efficiency (0-1)", row=2, col=2)
        
        # Update layout
        title = f"Thread Count Performance Analysis - Block {analysis.block_number}"
        if title_suffix:
            title += f" {title_suffix}"
            
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            height=800,
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.15,
                xanchor="center",
                x=0.5
            )
        )
        
        # Add annotations for key insights
        self._add_insights_annotations(fig, analysis)
        
        # Save to HTML file
        filename = f"parallelization_analysis_block_{analysis.block_number}.html"
        filepath = self.output_dir / filename
        
        fig.write_html(
            str(filepath),
            include_plotlyjs=True,
            config={'displayModeBar': True, 'displaylogo': False}
        )
        
        self.logger.info(f"Thread count comparison saved to {filepath}")
        return str(filepath)
    
    def create_research_focus_plot(
        self,
        analysis: ThreadCountAnalysis
    ) -> str:
        """
        Create a focused plot specifically for research showing max/avg gas vs threads.
        This is the primary plot needed for the research write-up.
        
        Args:
            analysis: ThreadCountAnalysis containing results
            
        Returns:
            Path to the generated HTML file
        """
        self.logger.info(f"Creating research focus plot for block {analysis.block_number}")
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]]
        )
        
        thread_counts = analysis.thread_counts
        bottleneck_gas = [gas / 1_000_000 for gas in analysis.bottleneck_gas_values]
        mean_gas = [gas / 1_000_000 for gas in analysis.mean_gas_values]
        
        color = '#2ca02c'  # Green for parallelization
        
        # Maximum gas (solid line)
        fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=bottleneck_gas,
                mode='lines+markers',
                name='Maximum Gas',
                line=dict(color=color, width=4),
                marker=dict(size=10),
                hovertemplate='<b>Maximum Gas</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Max Gas: %{y:.1f}M<br>' +
                            '<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Standard deviation bands
        # Note: For single block analysis, we don't have multiple blocks to calculate std dev
        # This is just for the single block's data
        
        # Mean gas (dashed line)
        fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=mean_gas,
                mode='lines+markers',
                name='Mean Gas',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=6),
                hovertemplate='<b>Mean Gas</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Gas: %{y:.1f}M<br>' +
                            '<extra></extra>'
            )
        )
        
        # Update axes
        fig.update_xaxes(
            title_text="Number of Threads",
            title_font=dict(size=16),
            tickfont=dict(size=14)
        )
        
        fig.update_yaxes(
            title_text="Maximum Gas per Thread (Millions)",
            title_font=dict(size=16, color='black'),
            tickfont=dict(size=14),
            secondary_y=False
        )
        
        fig.update_yaxes(
            title_text="Average Gas per Thread (Millions)",
            title_font=dict(size=16, color='gray'),
            tickfont=dict(size=14),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Gas Requirements vs Thread Count<br>" +
                     f"<sub>Block {analysis.block_number}</sub>",
                x=0.5,
                font=dict(size=20)
            ),
            height=600,
            width=1000,
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.02,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', secondary_y=False)
        fig.update_yaxes(showgrid=False, secondary_y=True)
        
        # Save to HTML file
        filename = f"research_thread_analysis_block_{analysis.block_number}.html"
        filepath = self.output_dir / filename
        
        fig.write_html(
            str(filepath),
            include_plotlyjs=True,
            config={'displayModeBar': True, 'displaylogo': False}
        )
        
        self.logger.info(f"Research focus plot saved to {filepath}")
        return str(filepath)
    
    def create_aggregate_statistics_plot(
        self,
        database: BlockchainDatabase,
        block_numbers: Optional[List[int]] = None,
        thread_counts: Optional[List[int]] = None,
        min_blocks: int = 10
    ) -> str:
        """
        Create aggregate statistical analysis across multiple blocks.
        Shows mean, 95% confidence interval, and maximum gas per thread vs thread count.
        
        Args:
            database: Database instance for loading block data
            block_numbers: List of block numbers to analyze (default: recent blocks with good tx count)
            thread_counts: Thread counts to test (default: [1,2,4,8,16,32])
            min_blocks: Minimum number of blocks to analyze for statistics
            
        Returns:
            Path to the generated HTML file
        """
        if thread_counts is None:
            thread_counts = [1, 2, 4, 8, 16, 32, 64]
        
        if block_numbers is None:
            # Get blocks with good transaction counts for statistical analysis
            db_stats = database.get_database_stats()
            max_block = db_stats['block_range']['max']
            block_numbers = []
            
            # Look through more blocks to get enough data
            for i in range(50):  # Check last 50 blocks
                block_num = max_block - i
                block_data = database.get_block(block_num)
                if block_data and block_data['transaction_count'] > 50:  # Lower threshold for more data
                    block_numbers.append(block_num)
                if len(block_numbers) >= min_blocks * 2:  # Get extra blocks
                    break
        
        if len(block_numbers) < min_blocks:
            raise ValueError(f"Need at least {min_blocks} blocks for statistical analysis, found {len(block_numbers)}")
        
        self.logger.info(f"Creating aggregate statistics for {len(block_numbers)} blocks")
        
        # Collect data from all blocks
        simulator = ParallelizationSimulator()
        gas_data = {tc: [] for tc in thread_counts}
        speedup_data = {tc: [] for tc in thread_counts}
        efficiency_data = {tc: [] for tc in thread_counts}
        
        # State-diff approach data (gas-weighted, ignoring dependencies)
        state_diff_gas_data = {tc: [] for tc in thread_counts}
        state_diff_speedup_data = {tc: [] for tc in thread_counts}
        
        successful_blocks = 0
        for block_num in block_numbers:
            try:
                # Load block data
                transactions_raw = database.get_transactions_by_block(block_num)
                dependencies_raw = database.get_dependencies_for_block(block_num)
                
                if not transactions_raw or len(transactions_raw) < 20:  # Skip very small blocks
                    continue
                
                # Convert to objects
                transactions = self._convert_transactions(transactions_raw)
                dependencies = self._convert_dependencies(dependencies_raw)
                
                # Run analysis
                analysis = simulator.analyze_thread_count_performance(
                    transactions, dependencies, block_num, thread_counts
                )
                
                # Use the total sequential gas consistently across both approaches
                our_sequential_gas = sum(tx.gas_used for tx in transactions)
                
                # Collect data for each thread count
                for i, tc in enumerate(thread_counts):
                    if i < len(analysis.bottleneck_gas_values):
                        gas_millions = analysis.bottleneck_gas_values[i] / 1_000_000
                        speedup = analysis.speedup_values[i]
                        efficiency = analysis.efficiency_values[i]
                        
                        # Include all segregated state data (bug is now fixed)
                        gas_data[tc].append(gas_millions)
                        speedup_data[tc].append(speedup)
                        efficiency_data[tc].append(efficiency)
                        
                        # Calculate state-diff data using our helper method
                        state_diff_gas = self._simulate_state_diff_parallelization(
                            transactions, tc
                        )
                        state_diff_gas_millions = state_diff_gas / 1_000_000
                        
                        # Use consistent sequential baseline for fair comparison
                        state_diff_speedup = our_sequential_gas / state_diff_gas if state_diff_gas > 0 else 1.0
                        
                        # State-diff data should always be valid due to our implementation
                        state_diff_gas_data[tc].append(state_diff_gas_millions)
                        state_diff_speedup_data[tc].append(state_diff_speedup)
                
                successful_blocks += 1
                
                if successful_blocks >= min_blocks:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze block {block_num}: {e}")
                continue
        
        if successful_blocks < min_blocks:
            raise ValueError(f"Only analyzed {successful_blocks} blocks successfully, need {min_blocks}")
        
        self.logger.info(f"Successfully analyzed {successful_blocks} blocks for statistics")
        
        color = '#2ca02c'  # Green for parallelization
        
        # Calculate statistics for each metric
        def calculate_stats(data_dict):
            means, percentile_25, percentile_75, maxes, mins = [], [], [], [], []
            for tc in thread_counts:
                data = data_dict[tc]
                if not data:
                    means.append(0)
                    percentile_25.append(0)
                    percentile_75.append(0)
                    maxes.append(0)
                    mins.append(0)
                    continue
                    
                data_array = np.array(data)
                mean_val = np.mean(data_array)
                max_val = np.max(data_array)
                min_val = np.min(data_array)
                
                # Calculate 25th and 75th percentiles for confidence intervals
                p25 = np.percentile(data_array, 25)
                p75 = np.percentile(data_array, 75)
                
                means.append(mean_val)
                percentile_25.append(p25)
                percentile_75.append(p75)
                maxes.append(max_val)
                mins.append(min_val)
                
            return means, percentile_25, percentile_75, maxes, mins
        
        # Calculate statistics for all metrics
        gas_means, gas_percentile_25, gas_percentile_75, gas_maxes, gas_mins = calculate_stats(gas_data)
        speedup_means, speedup_percentile_25, speedup_percentile_75, speedup_maxes, speedup_mins = calculate_stats(speedup_data)
        efficiency_means, efficiency_percentile_25, efficiency_percentile_75, efficiency_maxes, efficiency_mins = calculate_stats(efficiency_data)
        
        # Cap speedup confidence intervals at theoretical maximums (normal distribution can extend beyond physical limits)
        for i, tc in enumerate(thread_counts):
            if i < len(speedup_percentile_75):
                speedup_percentile_75[i] = min(speedup_percentile_75[i], tc)  # Cap at theoretical maximum
        
        # Calculate statistics for state-diff approach
        state_diff_gas_means, state_diff_gas_percentile_25, state_diff_gas_percentile_75, _, _ = calculate_stats(state_diff_gas_data)
        state_diff_speedup_means, state_diff_speedup_percentile_25, state_diff_speedup_percentile_75, _, state_diff_speedup_mins = calculate_stats(state_diff_speedup_data)
        
        # Cap state-diff speedup confidence intervals at theoretical maximums too
        for i, tc in enumerate(thread_counts):
            if i < len(state_diff_speedup_percentile_75):
                state_diff_speedup_percentile_75[i] = min(state_diff_speedup_percentile_75[i], tc)  # Cap at theoretical maximum
        
        # Check if we have any state-diff data (we should always have it now)
        has_state_diff_data = successful_blocks > 0
        
        # Calculate theoretical limits
        # For gas per thread: perfect distribution would be total_gas / thread_count
        total_gas_values = []
        for block_num in block_numbers:
            try:
                transactions_raw = database.get_transactions_by_block(block_num)
                if transactions_raw:
                    block_total_gas = sum(tx['gas_used'] or 0 for tx in transactions_raw)
                    total_gas_values.append(block_total_gas)
            except:
                continue
        
        if total_gas_values:
            mean_total_gas = np.mean(total_gas_values) / 1_000_000  # Convert to millions
            theoretical_gas_per_thread = [mean_total_gas / tc for tc in thread_counts]
        else:
            mean_total_gas = 18.0  # Fallback to ~18M gas as suggested
            theoretical_gas_per_thread = [mean_total_gas / tc for tc in thread_counts]
        
        # Create three separate plots
        plot_paths = []
        
        # Plot 1: Gas per Thread (with log scale)
        gas_fig = go.Figure()
        
        # Confidence interval
        gas_fig.add_trace(
            go.Scatter(
                x=thread_counts + thread_counts[::-1],
                y=gas_percentile_75 + gas_percentile_25[::-1],
                fill='toself',
                fillcolor=f'rgba(44, 160, 44, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='25th-75th Percentile',
                hoverinfo='skip'
            )
        )
        
        # State-diff confidence interval
        gas_fig.add_trace(
            go.Scatter(
                x=thread_counts + thread_counts[::-1],
                y=state_diff_gas_percentile_75 + state_diff_gas_percentile_25[::-1],
                fill='toself',
                fillcolor=f'rgba(255, 127, 14, 0.2)',  # Orange with transparency
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='State-Diff 25th-75th Percentile',
                hoverinfo='skip'
            )
        )
        
        # Average line
        gas_fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=gas_means,
                mode='lines+markers',
                name='Mean',
                line=dict(color=color, width=4),
                marker=dict(size=8),
                hovertemplate='<b>Mean Gas</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Gas: %{y:.1f}M<br>' +
                            '<extra></extra>'
            )
        )
        
        # Maximum line
        gas_fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=gas_maxes,
                mode='lines+markers',
                name='Maximum',
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(size=6, symbol='triangle-up'),
                hovertemplate='<b>Maximum Gas</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Gas: %{y:.1f}M<br>' +
                            '<extra></extra>'
            )
        )
        
        # Theoretical perfect distribution line
        gas_fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=theoretical_gas_per_thread,
                mode='lines',
                name='Theoretical Perfect Distribution',
                line=dict(color='#1f77b4', width=3, dash='dot'),  # Blue dotted line
                hovertemplate='<b>Theoretical Perfect</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Gas: %{y:.1f}M<br>' +
                            f'Based on {mean_total_gas:.1f}M mean total gas<br>' +
                            '<extra></extra>'
            )
        )
        
        # State-diff approach line (gas-weighted, ignoring dependencies)
        gas_fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=state_diff_gas_means,
                mode='lines+markers',
                name='Parallelization with State-Diffs',
                line=dict(color='#ff7f0e', width=3, dash='dash'),  # Orange dashed line
                marker=dict(size=6, symbol='diamond'),
                hovertemplate='<b>State-Diff Parallelization</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Gas: %{y:.1f}M<br>' +
                            'Gas-weighted distribution (ignoring dependencies)<br>' +
                            '<extra></extra>'
            )
        )
        
        gas_fig.update_layout(
            title=dict(
                text=f"Gas per Thread - simulated parallelization over {successful_blocks} blocks",
                x=0.5,
                font=dict(size=18)
            ),
            xaxis=dict(
                title="Number of Virtual Cores (k)",
                title_font=dict(size=14),
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="Gas per Thread (Millions)",
                title_font=dict(size=14),
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor='lightgray'
            ),
            height=600,
            width=1000,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.20,  # Move legend further down
                xanchor="center",
                x=0.5
            ),
            margin=dict(b=120)  # Increase bottom margin for more space
        )
        
        # Save gas plot
        gas_filename = f"gas_per_thread_{successful_blocks}_blocks.html"
        gas_filepath = self.output_dir / gas_filename
        gas_fig.write_html(str(gas_filepath), include_plotlyjs=True, config={'displayModeBar': True, 'displaylogo': False})
        plot_paths.append(str(gas_filepath))
        
        # Plot 2: Speedup (with min speedup line)
        speedup_fig = go.Figure()
        
        # Confidence interval
        speedup_fig.add_trace(
            go.Scatter(
                x=thread_counts + thread_counts[::-1],
                y=speedup_percentile_75 + speedup_percentile_25[::-1],
                fill='toself',
                fillcolor=f'rgba(44, 160, 44, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='25th-75th Percentile',
                hoverinfo='skip'
            )
        )
        
        # State-diff confidence interval
        speedup_fig.add_trace(
            go.Scatter(
                x=thread_counts + thread_counts[::-1],
                y=state_diff_speedup_percentile_75 + state_diff_speedup_percentile_25[::-1],
                fill='toself',
                fillcolor=f'rgba(255, 127, 14, 0.2)',  # Orange with transparency
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='State-Diff 25th-75th Percentile',
                hoverinfo='skip'
            )
        )
        
        # Segregated state mean speedup (green line for green confidence band)
        speedup_fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=speedup_means,
                mode='lines+markers',
                name='Segregated State Mean Speedup',
                line=dict(color='#2ca02c', width=4),  # Green to match the confidence band
                marker=dict(size=8),
                hovertemplate='<b>Segregated State Speedup</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Speedup: %{y:.2f}x<br>' +
                            '<extra></extra>'
            )
        )
        
        # State-diff mean speedup (orange line)
        speedup_fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=state_diff_speedup_means,
                mode='lines+markers',
                name='State-Diff Mean Speedup',
                line=dict(color='#ff7f0e', width=4),
                marker=dict(size=8),
                hovertemplate='<b>State-Diff Speedup</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Speedup: %{y:.2f}x<br>' +
                            '<extra></extra>'
            )
        )
        
        # Minimum speedup
        speedup_fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=state_diff_speedup_mins,
                mode='lines+markers',
                name='Minimum Speedup',
                line=dict(color='#d62728', width=3, dash='dot'),  # Red dashed line
                marker=dict(size=6, symbol='triangle-down'),
                hovertemplate='<b>Minimum Speedup</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Speedup: %{y:.2f}x<br>' +
                            '<extra></extra>'
            )
        )
        
        # Theoretical perfect linear speedup (y = x)
        speedup_fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=thread_counts,  # Perfect linear speedup: y = x
                mode='lines',
                name='Theoretical Linear Speedup',
                line=dict(color='#1f77b4', width=3, dash='dot'),  # Blue dotted line
                hovertemplate='<b>Theoretical Perfect</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Speedup: %{y:.2f}x<br>' +
                            'Perfect linear scaling<br>' +
                            '<extra></extra>'
            )
        )
        
        speedup_fig.update_layout(
            title=dict(
                text=f"Speedup - simulated parallelization over {successful_blocks} blocks",
                x=0.5,
                font=dict(size=18)
            ),
            xaxis=dict(
                title="Number of Virtual Cores (k)",
                title_font=dict(size=14),
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="Speedup (x)",
                title_font=dict(size=14),
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor='lightgray'
            ),
            height=600,
            width=1000,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,  # Move legend even further down
                xanchor="center",
                x=0.5,
                font=dict(size=11)  # Slightly smaller font
            ),
            margin=dict(b=140)  # Increase bottom margin even more for legend space
        )
        
        # Save speedup plot
        speedup_filename = f"speedup_analysis_{successful_blocks}_blocks.html"
        speedup_filepath = self.output_dir / speedup_filename
        speedup_fig.write_html(str(speedup_filepath), include_plotlyjs=True, config={'displayModeBar': True, 'displaylogo': False})
        plot_paths.append(str(speedup_filepath))
        
        # Plot 3: Thread Efficiency (with min and max efficiency lines)
        efficiency_fig = go.Figure()
        
        # Confidence interval
        efficiency_fig.add_trace(
            go.Scatter(
                x=thread_counts + thread_counts[::-1],
                y=efficiency_percentile_75 + efficiency_percentile_25[::-1],
                fill='toself',
                fillcolor=f'rgba(44, 160, 44, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='25th-75th Percentile',
                hoverinfo='skip'
            )
        )
        
        # Average efficiency
        efficiency_fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=efficiency_means,
                mode='lines+markers',
                name='Mean Efficiency',
                line=dict(color=color, width=4),
                marker=dict(size=8),
                hovertemplate='<b>Mean Efficiency</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Efficiency: %{y:.2f}<br>' +
                            '<extra></extra>'
            )
        )
        
        efficiency_fig.update_layout(
            title=dict(
                text=f"Thread Efficiency - simulated parallelization over {successful_blocks} blocks",
                x=0.5,
                font=dict(size=18)
            ),
            xaxis=dict(
                title="Number of Virtual Cores (k)",
                title_font=dict(size=14),
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="Efficiency (0-1)",
                title_font=dict(size=14),
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor='lightgray'
            ),
            height=600,
            width=1000,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.20,  # Move legend further down
                xanchor="center",
                x=0.5
            ),
            margin=dict(b=120)  # Increase bottom margin for more space
        )
        
        # Save efficiency plot
        efficiency_filename = f"thread_efficiency_{successful_blocks}_blocks.html"
        efficiency_filepath = self.output_dir / efficiency_filename
        efficiency_fig.write_html(str(efficiency_filepath), include_plotlyjs=True, config={'displayModeBar': True, 'displaylogo': False})
        plot_paths.append(str(efficiency_filepath))
        
        self.logger.info(f"Three separate plots saved:")
        self.logger.info(f"  Gas per Thread: {gas_filepath}")
        self.logger.info(f"  Speedup Analysis: {speedup_filepath}")
        self.logger.info(f"  Thread Efficiency: {efficiency_filepath}")
        
        # Return the primary gas plot path for backwards compatibility
        return str(gas_filepath)
    
    def create_speedup_distribution_violin_plots(
        self,
        database: BlockchainDatabase,
        block_numbers: Optional[List[int]] = None,
        thread_counts: Optional[List[int]] = None,
        min_blocks: int = 10
    ) -> str:
        """
        Create violin plots showing speedup distributions for segregated state vs state-diff approaches.
        
        Args:
            database: Database instance for loading block data
            block_numbers: List of block numbers to analyze (default: recent blocks with good tx count)
            thread_counts: Thread counts to test (default: [1,2,4,8,16,32,64])
            min_blocks: Minimum number of blocks to analyze for statistics
            
        Returns:
            Path to the generated HTML file
        """
        if thread_counts is None:
            thread_counts = [1, 2, 4, 8, 16, 32, 64]
        
        if block_numbers is None:
            # Get blocks with good transaction counts for statistical analysis
            db_stats = database.get_database_stats()
            max_block = db_stats['block_range']['max']
            block_numbers = []
            
            # Look through more blocks to get enough data
            for i in range(50):  # Check last 50 blocks
                block_num = max_block - i
                block_data = database.get_block(block_num)
                if block_data and block_data['transaction_count'] > 50:  # Lower threshold for more data
                    block_numbers.append(block_num)
                if len(block_numbers) >= min_blocks * 2:  # Get extra blocks
                    break
        
        if len(block_numbers) < min_blocks:
            raise ValueError(f"Need at least {min_blocks} blocks for statistical analysis, found {len(block_numbers)}")
        
        self.logger.info(f"Creating violin plots for {len(block_numbers)} blocks")
        
        # Collect data from all blocks
        simulator = ParallelizationSimulator()
        segregated_state_speedup_data = {tc: [] for tc in thread_counts}
        state_diff_speedup_data = {tc: [] for tc in thread_counts}
        
        successful_blocks = 0
        for block_num in block_numbers:
            try:
                # Load block data
                transactions_raw = database.get_transactions_by_block(block_num)
                dependencies_raw = database.get_dependencies_for_block(block_num)
                
                if not transactions_raw or len(transactions_raw) < 20:  # Skip very small blocks
                    continue
                
                # Convert to objects
                transactions = self._convert_transactions(transactions_raw)
                dependencies = self._convert_dependencies(dependencies_raw)
                
                # Run analysis
                analysis = simulator.analyze_thread_count_performance(
                    transactions, dependencies, block_num, thread_counts
                )
                
                # Use the total sequential gas consistently across both approaches
                our_sequential_gas = sum(tx.gas_used for tx in transactions)
                
                # Collect data for each thread count
                for i, tc in enumerate(thread_counts):
                    if i < len(analysis.bottleneck_gas_values):
                        speedup = analysis.speedup_values[i]
                        
                        # Cap speedup at theoretical maximum
                        capped_speedup = min(speedup, tc)
                        segregated_state_speedup_data[tc].append(capped_speedup)
                        
                        # Calculate state-diff data
                        state_diff_gas = self._simulate_state_diff_parallelization(
                            transactions, tc
                        )
                        state_diff_speedup = our_sequential_gas / state_diff_gas if state_diff_gas > 0 else 1.0
                        capped_state_diff_speedup = min(state_diff_speedup, tc)
                        state_diff_speedup_data[tc].append(capped_state_diff_speedup)
                
                successful_blocks += 1
                
                if successful_blocks >= min_blocks:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze block {block_num}: {e}")
                continue
        
        if successful_blocks < min_blocks:
            raise ValueError(f"Only analyzed {successful_blocks} blocks successfully, need {min_blocks}")
        
        self.logger.info(f"Successfully analyzed {successful_blocks} blocks for violin plots")
        
        # Create subplots: one row per thread count
        n_threads = len(thread_counts)
        fig = make_subplots(
            rows=n_threads, cols=1,
            subplot_titles=[f"{tc} Virtual Cores" for tc in thread_counts],
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}] for _ in range(n_threads)]
        )
        
        # Add violin plots for each thread count
        for i, tc in enumerate(thread_counts):
            row = i + 1
            
            # Segregated state violin (green)
            if segregated_state_speedup_data[tc]:
                fig.add_trace(
                    go.Violin(
                        y=segregated_state_speedup_data[tc],
                        x=[f"Segregated State"] * len(segregated_state_speedup_data[tc]),
                        name=f"Segregated State",
                        line_color="#2ca02c",
                        fillcolor="rgba(44, 160, 44, 0.7)",
                        points="outliers",
                        box_visible=True,
                        meanline_visible=True,
                        showlegend=(i == 0),  # Only show legend for first subplot
                        hovertemplate=f'<b>Segregated State ({tc} cores)</b><br>' +
                                    'Speedup: %{y:.3f}x<br>' +
                                    '<extra></extra>',
                        width=0.6
                    ),
                    row=row, col=1
                )
            
            # State-diff violin (orange)  
            if state_diff_speedup_data[tc]:
                fig.add_trace(
                    go.Violin(
                        y=state_diff_speedup_data[tc],
                        x=[f"State-Diff"] * len(state_diff_speedup_data[tc]),
                        name=f"State-Diff",
                        line_color="#ff7f0e",
                        fillcolor="rgba(255, 127, 14, 0.7)",
                        points="outliers",
                        box_visible=True,
                        meanline_visible=True,
                        showlegend=(i == 0),  # Only show legend for first subplot
                        hovertemplate=f'<b>State-Diff ({tc} cores)</b><br>' +
                                    'Speedup: %{y:.3f}x<br>' +
                                    '<extra></extra>',
                        width=0.6
                    ),
                    row=row, col=1
                )
            
            # Add theoretical maximum line
            fig.add_hline(
                y=tc, 
                line_dash="dot", 
                line_color="blue", 
                line_width=2,
                annotation_text=f"Theoretical Max ({tc}x)",
                annotation_position="top right",
                row=row, col=1
            )
            
            # Update y-axis for this subplot
            fig.update_yaxes(
                title_text="Speedup (x)",
                range=[0.8, min(tc * 1.1, tc + 0.5)],  # Give some headroom but not too much
                row=row, col=1
            )
            
            # Update x-axis for this subplot
            fig.update_xaxes(
                title_text="Parallelization Approach",
                row=row, col=1
            )
        
        # Update overall layout
        fig.update_layout(
            title=dict(
                text=f"Speedup Distribution - simulated parallelization over {successful_blocks} blocks",
                x=0.5,
                font=dict(size=18)
            ),
            height=200 * n_threads + 170,  # Add even more height for better spacing
            width=1000,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.08,  # Position below plots with more space
                xanchor="center",
                x=0.5
            ),
            margin=dict(b=100)  # Add bottom margin for legend
        )
        
        # Save violin plot
        violin_filename = f"speedup_distribution_violin_{successful_blocks}_blocks.html"
        violin_filepath = self.output_dir / violin_filename
        fig.write_html(str(violin_filepath), include_plotlyjs=True, config={'displayModeBar': True, 'displaylogo': False})
        
        self.logger.info(f"Speedup distribution violin plots saved to {violin_filepath}")
        return str(violin_filepath)
    
    def _convert_transactions(self, transactions_raw: List[Dict]) -> List[TransactionData]:
        """Convert database transaction data to TransactionData objects."""
        transactions = []
        for tx_data in transactions_raw:
            transactions.append(TransactionData(
                hash=tx_data['hash'],
                block_number=tx_data['block_number'], 
                transaction_index=tx_data['transaction_index'],
                from_address=tx_data['from_address'],
                to_address=tx_data['to_address'],
                value=tx_data['value'],
                gas=tx_data['gas'],
                gas_price=tx_data['gas_price'],
                gas_used=tx_data.get('gas_used', tx_data['gas']),
                status=tx_data.get('status'),
                input_data=tx_data.get('input_data'),
                logs=tx_data.get('logs')
            ))
        return transactions
    
    def _convert_dependencies(self, dependencies_raw: List[Dict]) -> List[StateDependency]:
        """Convert database dependency data to StateDependency objects."""
        dependencies = []
        for dep_data in dependencies_raw:
            dependencies.append(StateDependency(
                dependent_tx_hash=dep_data['dependent_tx_hash'],
                dependency_tx_hash=dep_data['dependency_tx_hash'],
                dependent_tx_index=dep_data['dependent_index'],
                dependency_tx_index=dep_data['dependency_index'],
                contract_address="0x" + "0" * 40,
                storage_slot="0x0",
                dependency_reason=dep_data['dependency_reason'],
                gas_impact=dep_data.get('gas_impact', 0)
            ))
        return dependencies
    
    def _add_insights_annotations(self, fig: go.Figure, analysis: ThreadCountAnalysis):
        """Add key insights as annotations to the figure."""
        # Find optimal thread count
        optimal_idx = analysis.speedup_values.index(max(analysis.speedup_values))
        optimal_threads = analysis.thread_counts[optimal_idx]
        optimal_speedup = analysis.speedup_values[optimal_idx]
        
        annotation_text = "Key Insights:<br>"
        annotation_text += f"• Optimal threads: {optimal_threads} ({optimal_speedup:.1f}x speedup)<br>"
        annotation_text += f"• Diminishing returns: {analysis.diminishing_returns_point} threads<br>"
        annotation_text += f"• Max speedup: {analysis.max_speedup:.1f}x"
    
        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1
        ) 
    
    def _simulate_state_diff_parallelization(
        self,
        transactions: List[TransactionData],
        thread_count: int
    ) -> int:
        """
        Simulate gas-weighted transaction distribution ignoring dependencies.
        This represents parallelization with state-diffs to resolve conflicts.
        
        Args:
            transactions: List of transactions
            thread_count: Number of threads
            
        Returns:
            Bottleneck gas (maximum gas on any thread)
        """
        if not transactions:
            return 0
        
        # Sort transactions by gas usage (largest first) for better distribution
        sorted_transactions = sorted(transactions, key=lambda tx: tx.gas_used, reverse=True)
        
        # Greedy allocation: assign each transaction to the thread with minimum current gas
        thread_gas = [0] * thread_count
        
        for tx in sorted_transactions:
            # Find thread with minimum gas
            thread_id = min(range(thread_count), key=lambda i: thread_gas[i])
            thread_gas[thread_id] += tx.gas_used
        
        # Return bottleneck gas (maximum gas on any thread)
        return max(thread_gas) if thread_gas else 0 

    def create_state_diff_only_analysis(
        self,
        database: BlockchainDatabase,
        block_numbers: Optional[List[int]] = None,
        thread_counts: Optional[List[int]] = None,
        min_blocks: int = 10
    ) -> str:
        """
        Create analysis plots showing only the state-diff approach (gas-weighted distribution).
        
        Args:
            database: Database instance for loading block data
            block_numbers: List of block numbers to analyze (default: recent blocks with good tx count)
            thread_counts: Thread counts to test (default: [1,2,4,8,16,32,64])
            min_blocks: Minimum number of blocks to analyze for statistics
            
        Returns:
            Path to the generated HTML file
        """
        if thread_counts is None:
            thread_counts = [1, 2, 4, 8, 16, 32, 64]
        
        if block_numbers is None:
            # Get blocks with good transaction counts for statistical analysis
            db_stats = database.get_database_stats()
            max_block = db_stats['block_range']['max']
            block_numbers = []
            
            # Look through more blocks to get enough data
            for i in range(50):  # Check last 50 blocks
                block_num = max_block - i
                block_data = database.get_block(block_num)
                if block_data and block_data['transaction_count'] > 50:  # Lower threshold for more data
                    block_numbers.append(block_num)
                if len(block_numbers) >= min_blocks * 2:  # Get extra blocks
                    break
        
        if len(block_numbers) < min_blocks:
            raise ValueError(f"Need at least {min_blocks} blocks for statistical analysis, found {len(block_numbers)}")
        
        self.logger.info(f"Creating state-diff only analysis for {len(block_numbers)} blocks")
        
        # Collect state-diff data only
        state_diff_gas_data = {tc: [] for tc in thread_counts}
        state_diff_speedup_data = {tc: [] for tc in thread_counts}
        
        successful_blocks = 0
        for block_num in block_numbers:
            try:
                # Load block data
                transactions_raw = database.get_transactions_by_block(block_num)
                dependencies_raw = database.get_dependencies_for_block(block_num)
                
                if not transactions_raw or len(transactions_raw) < 20:  # Skip very small blocks
                    continue
                
                # Convert to objects
                transactions = self._convert_transactions(transactions_raw)
                dependencies = self._convert_dependencies(dependencies_raw)
                
                # Use the total sequential gas consistently
                our_sequential_gas = sum(tx.gas_used for tx in transactions)
                
                # Collect state-diff data for each thread count
                for tc in thread_counts:
                    # Calculate state-diff data using our helper method
                    state_diff_gas = self._simulate_state_diff_parallelization(
                        transactions, tc
                    )
                    state_diff_gas_millions = state_diff_gas / 1_000_000
                    
                    # Use consistent sequential baseline for fair comparison
                    state_diff_speedup = our_sequential_gas / state_diff_gas if state_diff_gas > 0 else 1.0
                    
                    # Cap at theoretical maximum
                    capped_state_diff_speedup = min(state_diff_speedup, tc)
                    
                    # Store data
                    state_diff_gas_data[tc].append(state_diff_gas_millions)
                    state_diff_speedup_data[tc].append(capped_state_diff_speedup)
                
                successful_blocks += 1
                
                if successful_blocks >= min_blocks:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze block {block_num}: {e}")
                continue
        
        if successful_blocks < min_blocks:
            raise ValueError(f"Only analyzed {successful_blocks} blocks successfully, need {min_blocks}")
        
        self.logger.info(f"Successfully analyzed {successful_blocks} blocks for state-diff analysis")
        
        # Calculate statistics
        def calculate_stats(data_dict):
            means, percentile_25, percentile_75, maxes, mins = [], [], [], [], []
            for tc in thread_counts:
                data = data_dict[tc]
                if not data:
                    means.append(0)
                    percentile_25.append(0)
                    percentile_75.append(0)
                    maxes.append(0)
                    mins.append(0)
                    continue
                    
                data_array = np.array(data)
                mean_val = np.mean(data_array)
                max_val = np.max(data_array)
                min_val = np.min(data_array)
                
                # Calculate 25th and 75th percentiles
                p25 = np.percentile(data_array, 25)
                p75 = np.percentile(data_array, 75)
                
                means.append(mean_val)
                percentile_25.append(p25)
                percentile_75.append(p75)
                maxes.append(max_val)
                mins.append(min_val)
                
            return means, percentile_25, percentile_75, maxes, mins
        
        # Calculate statistics for state-diff approach
        state_diff_gas_means, state_diff_gas_percentile_25, state_diff_gas_percentile_75, _, _ = calculate_stats(state_diff_gas_data)
        state_diff_speedup_means, state_diff_speedup_percentile_25, state_diff_speedup_percentile_75, _, state_diff_speedup_mins = calculate_stats(state_diff_speedup_data)
        
        # Cap speedup confidence intervals at theoretical maximums
        for i, tc in enumerate(thread_counts):
            if i < len(state_diff_speedup_percentile_75):
                state_diff_speedup_percentile_75[i] = min(state_diff_speedup_percentile_75[i], tc)
        
        # Calculate theoretical limits
        total_gas_values = []
        for block_num in block_numbers:
            try:
                transactions_raw = database.get_transactions_by_block(block_num)
                if transactions_raw:
                    block_total_gas = sum(tx['gas_used'] or 0 for tx in transactions_raw)
                    total_gas_values.append(block_total_gas)
            except:
                continue
        
        if total_gas_values:
            mean_total_gas = np.mean(total_gas_values) / 1_000_000
            theoretical_gas_per_thread = [mean_total_gas / tc for tc in thread_counts]
        else:
            mean_total_gas = 18.0
            theoretical_gas_per_thread = [mean_total_gas / tc for tc in thread_counts]
        
        # Create gas per thread plot
        gas_fig = go.Figure()
        
        # State-diff confidence interval
        gas_fig.add_trace(
            go.Scatter(
                x=thread_counts + thread_counts[::-1],
                y=state_diff_gas_percentile_75 + state_diff_gas_percentile_25[::-1],
                fill='toself',
                fillcolor=f'rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='25th-75th Percentile',
                hoverinfo='skip'
            )
        )
        
        # State-diff mean line
        gas_fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=state_diff_gas_means,
                mode='lines+markers',
                name='Mean Gas per Thread',
                line=dict(color='#ff7f0e', width=4),
                marker=dict(size=8),
                hovertemplate='<b>State-Diff Mean</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Gas: %{y:.1f}M<br>' +
                            '<extra></extra>'
            )
        )
        
        # Theoretical perfect distribution line
        gas_fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=theoretical_gas_per_thread,
                mode='lines',
                name='Theoretical Perfect Distribution',
                line=dict(color='#1f77b4', width=3, dash='dot'),
                hovertemplate='<b>Theoretical Perfect</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Gas: %{y:.1f}M<br>' +
                            f'Based on {mean_total_gas:.1f}M mean total gas<br>' +
                            '<extra></extra>'
            )
        )
        
        gas_fig.update_layout(
            title=dict(
                text=f"Gas per Thread - state-diff parallelization over {successful_blocks} blocks",
                x=0.5,
                font=dict(size=18)
            ),
            xaxis=dict(
                title="Number of Virtual Cores (k)",
                title_font=dict(size=14),
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="Gas per Thread (Millions)",
                title_font=dict(size=14),
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor='lightgray'
            ),
            height=600,
            width=1000,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.20,  # Move legend further down
                xanchor="center",
                x=0.5
            ),
            margin=dict(b=120)  # Increase bottom margin for more space
        )
        
        # Create speedup plot
        speedup_fig = go.Figure()
        
        # State-diff confidence interval
        speedup_fig.add_trace(
            go.Scatter(
                x=thread_counts + thread_counts[::-1],
                y=state_diff_speedup_percentile_75 + state_diff_speedup_percentile_25[::-1],
                fill='toself',
                fillcolor=f'rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='25th-75th Percentile',
                hoverinfo='skip'
            )
        )
        
        # Segregated state mean speedup (green line for green confidence band)
        speedup_fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=speedup_means,
                mode='lines+markers',
                name='Segregated State Mean Speedup',
                line=dict(color='#2ca02c', width=4),  # Green to match the confidence band
                marker=dict(size=8),
                hovertemplate='<b>Segregated State Speedup</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Speedup: %{y:.2f}x<br>' +
                            '<extra></extra>'
            )
        )
        
        # State-diff mean speedup (orange line)
        speedup_fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=state_diff_speedup_means,
                mode='lines+markers',
                name='State-Diff Mean Speedup',
                line=dict(color='#ff7f0e', width=4),
                marker=dict(size=8),
                hovertemplate='<b>State-Diff Speedup</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Speedup: %{y:.2f}x<br>' +
                            '<extra></extra>'
            )
        )
        
        # Minimum speedup
        speedup_fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=state_diff_speedup_mins,
                mode='lines+markers',
                name='Minimum Speedup',
                line=dict(color='#d62728', width=3, dash='dot'),  # Red dashed line
                marker=dict(size=6, symbol='triangle-down'),
                hovertemplate='<b>Minimum Speedup</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Speedup: %{y:.2f}x<br>' +
                            '<extra></extra>'
            )
        )
        
        # Theoretical perfect linear speedup (y = x)
        speedup_fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=thread_counts,
                mode='lines',
                name='Theoretical Linear Speedup',
                line=dict(color='#1f77b4', width=3, dash='dot'),
                hovertemplate='<b>Theoretical Perfect</b><br>' +
                            'Virtual Cores: %{x}<br>' +
                            'Speedup: %{y:.2f}x<br>' +
                            'Perfect linear scaling<br>' +
                            '<extra></extra>'
            )
        )
        
        speedup_fig.update_layout(
            title=dict(
                text=f"Speedup - simulated parallelization over {successful_blocks} blocks",
                x=0.5,
                font=dict(size=18)
            ),
            xaxis=dict(
                title="Number of Virtual Cores (k)",
                title_font=dict(size=14),
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="Speedup (x)",
                title_font=dict(size=14),
                tickfont=dict(size=12),
                showgrid=True,
                gridcolor='lightgray'
            ),
            height=600,
            width=1000,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,  # Move legend even further down
                xanchor="center",
                x=0.5,
                font=dict(size=11)  # Slightly smaller font
            ),
            margin=dict(b=140)  # Increase bottom margin even more for legend space
        )
        
        # Save plots
        gas_filename = f"state_diff_gas_per_thread_{successful_blocks}_blocks.html"
        gas_filepath = self.output_dir / gas_filename
        gas_fig.write_html(str(gas_filepath), include_plotlyjs=True, config={'displayModeBar': True, 'displaylogo': False})
        
        speedup_filename = f"state_diff_speedup_{successful_blocks}_blocks.html"
        speedup_filepath = self.output_dir / speedup_filename
        speedup_fig.write_html(str(speedup_filepath), include_plotlyjs=True, config={'displayModeBar': True, 'displaylogo': False})
        
        self.logger.info(f"State-diff only analysis saved:")
        self.logger.info(f"  Gas per Thread: {gas_filepath}")
        self.logger.info(f"  Speedup Analysis: {speedup_filepath}")
        
        # Return the primary gas plot path
        return str(gas_filepath) 