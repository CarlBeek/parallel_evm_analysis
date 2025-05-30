"""
Parallelization comparison visualization for thread count performance analysis.
Creates interactive plots showing how gas requirements change with thread count across different strategies.
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
    ParallelizationStrategy,
    MultiStrategyAnalysis,
    ThreadCountAnalysis
)
from storage.database import BlockchainDatabase
from core.transaction_fetcher import TransactionData
from analysis.state_dependency_analyzer import StateDependency


class ParallelizationComparisonVisualizer:
    """
    Creates interactive visualizations comparing parallelization strategies across thread counts.
    Generates the research plots needed for thread count vs. gas requirements analysis.
    """
    
    def __init__(self, output_dir: str = "./data/graphs"):
        """
        Initialize the parallelization comparison visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Color scheme for strategies
        self.strategy_colors = {
            ParallelizationStrategy.SEQUENTIAL: '#1f77b4',      # Blue
            ParallelizationStrategy.DEPENDENCY_AWARE: '#2ca02c'  # Green
        }
    
    def create_thread_count_comparison(
        self,
        analysis: MultiStrategyAnalysis,
        title_suffix: str = ""
    ) -> str:
        """
        Create comprehensive thread count comparison visualization.
        
        Args:
            analysis: MultiStrategyAnalysis containing all strategy results
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
                'Average Gas per Thread (Utilization)', 
                'Parallel Speedup vs Sequential',
                'Thread Efficiency (Load Balance)'
            ],
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Process data for each strategy
        for strategy, strategy_analysis in analysis.strategy_analyses.items():
            color = self.strategy_colors[strategy]
            name = strategy.value.replace('_', ' ').title()
            
            thread_counts = strategy_analysis.thread_counts
            bottleneck_gas = strategy_analysis.bottleneck_gas_values
            average_gas = strategy_analysis.average_gas_values
            speedup = strategy_analysis.speedup_values
            efficiency = strategy_analysis.efficiency_values
            
            # Plot 1: Bottleneck Gas (most important for research)
            fig.add_trace(
                go.Scatter(
                    x=thread_counts,
                    y=[gas / 1_000_000 for gas in bottleneck_gas],  # Convert to millions
                    mode='lines+markers',
                    name=name,
                    line=dict(color=color, width=3),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{name}</b><br>' +
                                'Threads: %{x}<br>' +
                                'Max Gas: %{y:.1f}M<br>' +
                                '<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Plot 2: Average Gas
            fig.add_trace(
                go.Scatter(
                    x=thread_counts,
                    y=[gas / 1_000_000 for gas in average_gas],
                    mode='lines+markers',
                    name=name,
                    line=dict(color=color, width=3),
                    marker=dict(size=8),
                    showlegend=False,
                    hovertemplate=f'<b>{name}</b><br>' +
                                'Threads: %{x}<br>' +
                                'Avg Gas: %{y:.1f}M<br>' +
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
                    name=name,
                    line=dict(color=color, width=3),
                    marker=dict(size=8),
                    showlegend=False,
                    hovertemplate=f'<b>{name}</b><br>' +
                                'Threads: %{x}<br>' +
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
                    name=name,
                    line=dict(color=color, width=3),
                    marker=dict(size=8),
                    showlegend=False,
                    hovertemplate=f'<b>{name}</b><br>' +
                                'Threads: %{x}<br>' +
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
        title = f"Parallelization Strategy Comparison - Block {analysis.block_number}"
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
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add annotations for key insights
        self._add_insights_annotations(fig, analysis)
        
        # Save to HTML file
        filename = f"parallelization_comparison_block_{analysis.block_number}.html"
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
        analysis: MultiStrategyAnalysis,
        strategies: Optional[List[ParallelizationStrategy]] = None
    ) -> str:
        """
        Create a focused plot specifically for research showing max/avg gas vs threads.
        This is the primary plot needed for the research write-up.
        
        Args:
            analysis: MultiStrategyAnalysis containing strategy results
            strategies: Optional list of strategies to include (default: all)
            
        Returns:
            Path to the generated HTML file
        """
        if strategies is None:
            strategies = list(analysis.strategy_analyses.keys())
        
        self.logger.info(f"Creating research focus plot for {len(strategies)} strategies")
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]]
        )
        
        # Plot maximum gas (primary y-axis) and average gas (secondary y-axis)
        for strategy in strategies:
            if strategy not in analysis.strategy_analyses:
                continue
                
            strategy_analysis = analysis.strategy_analyses[strategy]
            color = self.strategy_colors[strategy]
            name = strategy.value.replace('_', ' ').title()
            
            thread_counts = strategy_analysis.thread_counts
            bottleneck_gas = [gas / 1_000_000 for gas in strategy_analysis.bottleneck_gas_values]
            average_gas = [gas / 1_000_000 for gas in strategy_analysis.average_gas_values]
            
            # Maximum gas (solid line)
            fig.add_trace(
                go.Scatter(
                    x=thread_counts,
                    y=bottleneck_gas,
                    mode='lines+markers',
                    name=f'{name} (Max)',
                    line=dict(color=color, width=4),
                    marker=dict(size=10),
                    hovertemplate=f'<b>{name} - Maximum Gas</b><br>' +
                                'Threads: %{x}<br>' +
                                'Max Gas: %{y:.1f}M<br>' +
                                '<extra></extra>'
                ),
                secondary_y=False
            )
            
            # Average gas (dashed line)
            fig.add_trace(
                go.Scatter(
                    x=thread_counts,
                    y=average_gas,
                    mode='lines+markers',
                    name=f'{name} (Avg)',
                    line=dict(color=color, width=3, dash='dash'),
                    marker=dict(size=8, symbol='diamond'),
                    hovertemplate=f'<b>{name} - Average Gas</b><br>' +
                                'Threads: %{x}<br>' +
                                'Avg Gas: %{y:.1f}M<br>' +
                                '<extra></extra>'
                ),
                secondary_y=True
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
                     f"<sub>Block {analysis.block_number} " +
                     f"({analysis.transaction_count:,} transactions, " +
                     f"{analysis.dependency_count} dependencies)</sub>",
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
    
    def create_multi_block_analysis(
        self,
        database: BlockchainDatabase,
        block_numbers: Optional[List[int]] = None,
        thread_counts: Optional[List[int]] = None,
        strategies: Optional[List[ParallelizationStrategy]] = None
    ) -> str:
        """
        Create analysis across multiple blocks to show consistency of results.
        
        Args:
            database: Database instance for loading block data
            block_numbers: List of block numbers to analyze (default: recent blocks)
            thread_counts: Thread counts to test (default: [1,2,4,8,16,32])
            strategies: Strategies to test (default: all)
            
        Returns:
            Path to the generated HTML file
        """
        if thread_counts is None:
            thread_counts = [1, 2, 4, 8, 16, 32]
        
        if strategies is None:
            strategies = [
                ParallelizationStrategy.SEQUENTIAL,
                ParallelizationStrategy.DEPENDENCY_AWARE
            ]
        
        if block_numbers is None:
            # Get 3 recent blocks with good transaction counts
            stats = database.get_database_stats()
            max_block = stats['block_range']['max']
            block_numbers = []
            
            for i in range(10):  # Check last 10 blocks
                block_num = max_block - i
                block_data = database.get_block(block_num)
                if block_data and block_data['transaction_count'] > 100:
                    block_numbers.append(block_num)
                if len(block_numbers) >= 3:
                    break
        
        self.logger.info(f"Creating multi-block analysis for {len(block_numbers)} blocks")
        
        # Analyze each block
        simulator = ParallelizationSimulator()
        all_analyses = []
        
        for block_num in block_numbers:
            try:
                # Load block data
                transactions_raw = database.get_transactions_by_block(block_num)
                dependencies_raw = database.get_dependencies_for_block(block_num)
                
                # Convert to objects
                transactions = self._convert_transactions(transactions_raw)
                dependencies = self._convert_dependencies(dependencies_raw)
                
                # Run analysis
                analysis = simulator.analyze_thread_count_performance(
                    transactions, dependencies, block_num, thread_counts, strategies
                )
                all_analyses.append(analysis)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze block {block_num}: {e}")
                continue
        
        if not all_analyses:
            raise ValueError("No blocks could be analyzed")
        
        # Create visualization
        fig = make_subplots(
            rows=len(strategies), cols=2,
            subplot_titles=[f'{strategy.value.replace("_", " ").title()} - Max Gas' 
                          for strategy in strategies] + 
                         [f'{strategy.value.replace("_", " ").title()} - Speedup' 
                          for strategy in strategies],
            vertical_spacing=0.08
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, analysis in enumerate(all_analyses):
            color = colors[i % len(colors)]
            block_label = f"Block {analysis.block_number}"
            
            for j, strategy in enumerate(strategies):
                if strategy in analysis.strategy_analyses:
                    strategy_analysis = analysis.strategy_analyses[strategy]
                    
                    # Max gas plot
                    fig.add_trace(
                        go.Scatter(
                            x=strategy_analysis.thread_counts,
                            y=[gas / 1_000_000 for gas in strategy_analysis.bottleneck_gas_values],
                            mode='lines+markers',
                            name=block_label,
                            line=dict(color=color),
                            showlegend=(j == 0),  # Only show legend for first strategy
                            hovertemplate=f'<b>{block_label}</b><br>' +
                                        'Threads: %{x}<br>' +
                                        'Max Gas: %{y:.1f}M<br>' +
                                        '<extra></extra>'
                        ),
                        row=j+1, col=1
                    )
                    
                    # Speedup plot
                    fig.add_trace(
                        go.Scatter(
                            x=strategy_analysis.thread_counts,
                            y=strategy_analysis.speedup_values,
                            mode='lines+markers',
                            name=block_label,
                            line=dict(color=color),
                            showlegend=False,
                            hovertemplate=f'<b>{block_label}</b><br>' +
                                        'Threads: %{x}<br>' +
                                        'Speedup: %{y:.2f}x<br>' +
                                        '<extra></extra>'
                        ),
                        row=j+1, col=2
                    )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Multi-Block Thread Count Analysis<br>" +
                     f"<sub>{len(all_analyses)} blocks compared</sub>",
                x=0.5,
                font=dict(size=20)
            ),
            height=300 * len(strategies),
            hovermode='x unified'
        )
        
        # Update axes
        for i in range(len(strategies)):
            fig.update_xaxes(title_text="Number of Threads", row=i+1, col=1)
            fig.update_xaxes(title_text="Number of Threads", row=i+1, col=2)
            fig.update_yaxes(title_text="Gas (Millions)", row=i+1, col=1)
            fig.update_yaxes(title_text="Speedup (x)", row=i+1, col=2)
        
        # Save to HTML file
        filename = f"multi_block_thread_analysis_{len(all_analyses)}_blocks.html"
        filepath = self.output_dir / filename
        
        fig.write_html(
            str(filepath),
            include_plotlyjs=True,
            config={'displayModeBar': True, 'displaylogo': False}
        )
        
        self.logger.info(f"Multi-block analysis saved to {filepath}")
        return str(filepath)
    
    def create_aggregate_statistics_plot(
        self,
        database: BlockchainDatabase,
        block_numbers: Optional[List[int]] = None,
        thread_counts: Optional[List[int]] = None,
        strategies: Optional[List[ParallelizationStrategy]] = None,
        min_blocks: int = 10
    ) -> str:
        """
        Create aggregate statistical analysis across multiple blocks.
        Shows average, 95% confidence interval, and maximum gas per thread vs thread count.
        
        Args:
            database: Database instance for loading block data
            block_numbers: List of block numbers to analyze (default: recent blocks with good tx count)
            thread_counts: Thread counts to test (default: [1,2,4,8,16,32])
            strategies: Strategies to test (default: all)
            min_blocks: Minimum number of blocks to analyze for statistics
            
        Returns:
            Path to the generated HTML file
        """
        if thread_counts is None:
            thread_counts = [1, 2, 4, 8, 16, 32]
        
        if strategies is None:
            strategies = [
                ParallelizationStrategy.SEQUENTIAL,
                ParallelizationStrategy.DEPENDENCY_AWARE
            ]
        
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
        strategy_data = {strategy: {tc: [] for tc in thread_counts} for strategy in strategies}
        
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
                    transactions, dependencies, block_num, thread_counts, strategies
                )
                
                # Collect bottleneck gas data for each strategy and thread count
                for strategy in strategies:
                    if strategy in analysis.strategy_analyses:
                        strategy_analysis = analysis.strategy_analyses[strategy]
                        for i, tc in enumerate(thread_counts):
                            if i < len(strategy_analysis.bottleneck_gas_values):
                                gas_millions = strategy_analysis.bottleneck_gas_values[i] / 1_000_000
                                strategy_data[strategy][tc].append(gas_millions)
                
                successful_blocks += 1
                
                if successful_blocks >= min_blocks:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze block {block_num}: {e}")
                continue
        
        if successful_blocks < min_blocks:
            raise ValueError(f"Only analyzed {successful_blocks} blocks successfully, need {min_blocks}")
        
        self.logger.info(f"Successfully analyzed {successful_blocks} blocks for statistics")
        
        # Calculate statistics for each strategy
        fig = go.Figure()
        
        for strategy in strategies:
            color = self.strategy_colors[strategy]
            name = strategy.value.replace('_', ' ').title()
            
            means = []
            ci_lower = []
            ci_upper = []
            maxes = []
            
            # Calculate statistics for each thread count
            for tc in thread_counts:
                data = strategy_data[strategy][tc]
                if not data:
                    continue
                    
                data_array = np.array(data)
                mean_val = np.mean(data_array)
                max_val = np.max(data_array)
                
                # Calculate 95% confidence interval
                if len(data) > 1:
                    sem = stats.sem(data_array)  # Standard error of mean
                    ci = stats.t.interval(0.95, len(data_array)-1, loc=mean_val, scale=sem)
                    ci_lower.append(ci[0])
                    ci_upper.append(ci[1])
                else:
                    ci_lower.append(mean_val)
                    ci_upper.append(mean_val)
                
                means.append(mean_val)
                maxes.append(max_val)
            
            # Plot confidence interval as filled area
            fig.add_trace(
                go.Scatter(
                    x=thread_counts + thread_counts[::-1],  # x, then x reversed
                    y=ci_upper + ci_lower[::-1],  # upper, then lower reversed
                    fill='toself',
                    fillcolor=f'rgba{tuple(list(plt_color_to_rgb(color)) + [0.2])}',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name=f'{name} 95% CI',
                    hoverinfo='skip'
                )
            )
            
            # Plot average line
            fig.add_trace(
                go.Scatter(
                    x=thread_counts,
                    y=means,
                    mode='lines+markers',
                    name=f'{name} (Average)',
                    line=dict(color=color, width=4),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{name} - Average</b><br>' +
                                'Threads: %{x}<br>' +
                                'Avg Gas: %{y:.1f}M<br>' +
                                f'Blocks: {successful_blocks}<br>' +
                                '<extra></extra>'
                )
            )
            
            # Plot maximum line
            fig.add_trace(
                go.Scatter(
                    x=thread_counts,
                    y=maxes,
                    mode='lines+markers',
                    name=f'{name} (Maximum)',
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=6, symbol='triangle-up'),
                    hovertemplate=f'<b>{name} - Maximum</b><br>' +
                                'Threads: %{x}<br>' +
                                'Max Gas: %{y:.1f}M<br>' +
                                f'Blocks: {successful_blocks}<br>' +
                                '<extra></extra>'
                )
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Aggregate Parallelization Statistics<br>" +
                     f"<sub>{successful_blocks} blocks analyzed - Average, 95% CI, and Maximum</sub>",
                x=0.5,
                font=dict(size=20)
            ),
            xaxis=dict(
                title="Number of Threads",
                title_font=dict(size=16),
                tickfont=dict(size=14),
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="Gas per Thread (Millions)",
                title_font=dict(size=16),
                tickfont=dict(size=14),
                showgrid=True,
                gridcolor='lightgray'
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
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add statistical summary annotation
        summary_text = f"Statistical Summary ({successful_blocks} blocks):<br>"
        for strategy in strategies:
            name = strategy.value.replace('_', ' ').title()
            # Calculate overall improvement
            if len(strategy_data[strategy][thread_counts[0]]) > 0 and len(strategy_data[strategy][thread_counts[-1]]) > 0:
                first_mean = np.mean(strategy_data[strategy][thread_counts[0]])
                last_mean = np.mean(strategy_data[strategy][thread_counts[-1]])
                improvement = first_mean / last_mean if last_mean > 0 else 1.0
                summary_text += f"• {name}: {improvement:.1f}x avg improvement<br>"
        
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            showarrow=False,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            xanchor="right",
            yanchor="top"
        )
        
        # Save to HTML file
        filename = f"aggregate_parallelization_statistics_{successful_blocks}_blocks.html"
        filepath = self.output_dir / filename
        
        fig.write_html(
            str(filepath),
            include_plotlyjs=True,
            config={'displayModeBar': True, 'displaylogo': False}
        )
        
        self.logger.info(f"Aggregate statistics plot saved to {filepath}")
        return str(filepath)
    
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
    
    def _add_insights_annotations(self, fig: go.Figure, analysis: MultiStrategyAnalysis):
        """Add key insights as annotations to the figure."""
        # Find the best performing strategy at different thread counts
        best_strategies = {}
        for thread_count, strategy in analysis.best_strategy_per_thread_count.items():
            if thread_count in [1, 4, 16]:  # Key thread counts
                best_strategies[thread_count] = strategy
        
        # Add annotation for key insight
        if best_strategies:
            annotation_text = "Key Insights:<br>"
            for tc, strategy in best_strategies.items():
                strategy_analysis = analysis.strategy_analyses[strategy]
                idx = strategy_analysis.thread_counts.index(tc)
                speedup = strategy_analysis.speedup_values[idx]
                annotation_text += f"• {tc} threads: {strategy.value.replace('_', ' ')} best ({speedup:.1f}x)<br>"
        
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

def plt_color_to_rgb(color_str: str) -> Tuple[int, int, int]:
    """Convert matplotlib color string to RGB tuple."""
    # Simple conversion for hex colors
    if color_str.startswith('#'):
        return tuple(int(color_str[i:i+2], 16) for i in (1, 3, 5))
    else:
        # Fallback for named colors - return a default
        return (128, 128, 128) 