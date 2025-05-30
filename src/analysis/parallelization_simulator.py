"""
Parallelization simulation framework for comparing different thread allocation strategies.
This module simulates how transactions would be distributed across N threads under various
parallelization approaches and calculates performance metrics.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple, NamedTuple
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import math
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.transaction_fetcher import TransactionData
from analysis.state_dependency_analyzer import StateDependency


class ParallelizationStrategy(Enum):
    """Enumeration of different parallelization strategies."""
    SEQUENTIAL = "sequential"  # Single-thread baseline
    DEPENDENCY_AWARE = "dependency_aware"  # Topological sorting with dependency chains


@dataclass
class TransactionAssignment:
    """Represents assignment of a transaction to a specific thread."""
    tx_hash: str
    tx_index: int
    thread_id: int
    gas_used: int
    execution_order: int  # Order within the thread


@dataclass
class ThreadMetrics:
    """Performance metrics for a single thread."""
    thread_id: int
    transaction_count: int
    total_gas: int
    execution_time_estimate: float  # Proportional to gas
    transactions: List[TransactionAssignment]
    conflicts: List[str] = None  # List of conflicting transaction hashes


@dataclass
class ParallelizationResult:
    """Complete result of a parallelization simulation."""
    strategy: ParallelizationStrategy
    thread_count: int
    block_number: int
    thread_metrics: List[ThreadMetrics]
    
    # Aggregate metrics
    bottleneck_gas: int  # Maximum gas on any thread (determines execution time)
    average_gas: float  # Mean gas across threads
    thread_efficiency: float  # Ratio of min/max gas (1.0 = perfect balance)
    conflict_rate: float  # Percentage of transactions with conflicts
    total_conflicts: int
    
    # Performance improvement metrics
    sequential_gas: int  # Total gas if executed sequentially
    parallel_speedup: float  # Sequential gas / bottleneck gas
    utilization_efficiency: float  # Average gas / bottleneck gas


@dataclass
class ThreadCountAnalysis:
    """Analysis results across multiple thread counts for a single strategy."""
    strategy: ParallelizationStrategy
    block_number: int
    thread_counts: List[int]
    bottleneck_gas_values: List[int]  # Max gas per thread count
    average_gas_values: List[float]  # Avg gas per thread count
    speedup_values: List[float]  # Speedup vs sequential per thread count
    efficiency_values: List[float]  # Thread efficiency per thread count
    
    # Analysis metrics
    optimal_thread_count: int  # Thread count with best speedup
    diminishing_returns_point: int  # Where improvement drops below threshold
    max_speedup: float  # Best speedup achieved
    efficiency_at_optimal: float  # Thread efficiency at optimal point


@dataclass
class MultiStrategyAnalysis:
    """Comprehensive analysis across all strategies and thread counts."""
    block_number: int
    total_gas: int
    transaction_count: int
    dependency_count: int
    
    # Per-strategy analysis
    strategy_analyses: Dict[ParallelizationStrategy, ThreadCountAnalysis]
    
    # Cross-strategy comparisons
    best_strategy_per_thread_count: Dict[int, ParallelizationStrategy]
    thread_count_recommendations: Dict[ParallelizationStrategy, int]


class DependencyGraph:
    """Represents transaction dependencies as a directed graph."""
    
    def __init__(self, transactions: List[TransactionData], dependencies: List[StateDependency]):
        """
        Initialize dependency graph.
        
        Args:
            transactions: List of all transactions in the block
            dependencies: List of state dependencies between transactions
        """
        self.transactions = {tx.hash: tx for tx in transactions}
        self.tx_by_index = {tx.transaction_index: tx for tx in transactions}
        
        # Build adjacency lists
        self.dependents = defaultdict(set)  # tx -> set of txs that depend on it
        self.dependencies_map = defaultdict(set)  # tx -> set of txs it depends on
        
        for dep in dependencies:
            self.dependencies_map[dep.dependent_tx_hash].add(dep.dependency_tx_hash)
            self.dependents[dep.dependency_tx_hash].add(dep.dependent_tx_hash)
    
    def get_dependency_chains(self) -> List[List[str]]:
        """Get all dependency chains in the graph."""
        visited = set()
        chains = []
        
        for tx_hash in self.transactions:
            if tx_hash not in visited and tx_hash not in self.dependencies_map:
                # This is a root transaction (no dependencies)
                chain = self._build_chain_from_root(tx_hash, visited)
                if chain:
                    chains.append(chain)
        
        return chains
    
    def _build_chain_from_root(self, root_tx: str, visited: set) -> List[str]:
        """Build dependency chain starting from a root transaction."""
        chain = []
        current = root_tx
        
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            
            # Find the next transaction in the chain
            dependents = self.dependents.get(current, set())
            if len(dependents) == 1:
                current = next(iter(dependents))
            elif len(dependents) > 1:
                # Multiple dependents - choose the one with highest gas
                current = max(dependents, key=lambda tx: self.transactions[tx].gas_used)
            else:
                current = None
        
        return chain
    
    def get_independent_transactions(self) -> List[str]:
        """Get transactions that have no dependencies and no dependents."""
        independent = []
        for tx_hash in self.transactions:
            if (tx_hash not in self.dependencies_map and 
                tx_hash not in self.dependents):
                independent.append(tx_hash)
        return independent
    
    def topological_sort(self) -> List[str]:
        """Return topologically sorted list of transactions."""
        # Kahn's algorithm
        in_degree = defaultdict(int)
        for tx_hash in self.transactions:
            in_degree[tx_hash] = len(self.dependencies_map[tx_hash])
        
        queue = [tx for tx in self.transactions if in_degree[tx] == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for dependent in self.dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return result


class ThreadCountPerformanceAnalyzer:
    """
    Specialized analyzer for studying how thread count affects performance.
    This provides the core research functionality for thread scalability analysis.
    """
    
    def __init__(self, simulator: 'ParallelizationSimulator'):
        """
        Initialize the thread count performance analyzer.
        
        Args:
            simulator: ParallelizationSimulator instance
        """
        self.simulator = simulator
        self.logger = logging.getLogger(__name__)
    
    def analyze_thread_scaling(
        self,
        transactions: List[TransactionData],
        dependencies: List[StateDependency],
        block_number: int,
        thread_counts: Optional[List[int]] = None,
        strategies: Optional[List[ParallelizationStrategy]] = None
    ) -> MultiStrategyAnalysis:
        """
        Analyze how different thread counts affect performance across strategies.
        
        Args:
            transactions: List of transactions to analyze
            dependencies: List of dependencies between transactions
            block_number: Block number being analyzed
            thread_counts: List of thread counts to test (default: [1,2,4,8,16,32,64])
            strategies: List of strategies to test (default: all strategies)
            
        Returns:
            Comprehensive analysis across all strategies and thread counts
        """
        if thread_counts is None:
            thread_counts = [1, 2, 4, 8, 16, 32, 64]
        
        if strategies is None:
            strategies = list(ParallelizationStrategy)
        
        self.logger.info(f"Analyzing thread scaling for block {block_number}: "
                        f"{len(thread_counts)} thread counts × {len(strategies)} strategies")
        
        # Analyze each strategy across all thread counts
        strategy_analyses = {}
        for strategy in strategies:
            analysis = self._analyze_strategy_scaling(
                strategy, transactions, dependencies, block_number, thread_counts
            )
            strategy_analyses[strategy] = analysis
        
        # Perform cross-strategy analysis
        best_per_thread = self._find_best_strategy_per_thread_count(strategy_analyses, thread_counts)
        recommendations = self._generate_thread_count_recommendations(strategy_analyses)
        
        return MultiStrategyAnalysis(
            block_number=block_number,
            total_gas=sum(tx.gas_used for tx in transactions),
            transaction_count=len(transactions),
            dependency_count=len(dependencies),
            strategy_analyses=strategy_analyses,
            best_strategy_per_thread_count=best_per_thread,
            thread_count_recommendations=recommendations
        )
    
    def _analyze_strategy_scaling(
        self,
        strategy: ParallelizationStrategy,
        transactions: List[TransactionData],
        dependencies: List[StateDependency],
        block_number: int,
        thread_counts: List[int]
    ) -> ThreadCountAnalysis:
        """Analyze a single strategy across multiple thread counts."""
        self.logger.info(f"Analyzing {strategy.value} across {len(thread_counts)} thread counts")
        
        bottleneck_values = []
        average_values = []
        speedup_values = []
        efficiency_values = []
        
        # Get sequential baseline for speedup calculation
        sequential_result = self.simulator.simulate_strategy(
            strategy, 1, transactions, dependencies, block_number
        )
        sequential_gas = sequential_result.sequential_gas
        
        # Test each thread count
        for thread_count in thread_counts:
            result = self.simulator.simulate_strategy(
                strategy, thread_count, transactions, dependencies, block_number
            )
            
            bottleneck_values.append(result.bottleneck_gas)
            average_values.append(result.average_gas)
            speedup_values.append(sequential_gas / result.bottleneck_gas if result.bottleneck_gas > 0 else 1.0)
            efficiency_values.append(result.thread_efficiency)
        
        # Find optimal thread count and diminishing returns point
        optimal_idx = speedup_values.index(max(speedup_values))
        optimal_thread_count = thread_counts[optimal_idx]
        max_speedup = speedup_values[optimal_idx]
        
        # Find diminishing returns point (where improvement drops below 10%)
        diminishing_point = self._find_diminishing_returns_point(speedup_values, thread_counts)
        
        return ThreadCountAnalysis(
            strategy=strategy,
            block_number=block_number,
            thread_counts=thread_counts,
            bottleneck_gas_values=bottleneck_values,
            average_gas_values=average_values,
            speedup_values=speedup_values,
            efficiency_values=efficiency_values,
            optimal_thread_count=optimal_thread_count,
            diminishing_returns_point=diminishing_point,
            max_speedup=max_speedup,
            efficiency_at_optimal=efficiency_values[optimal_idx]
        )
    
    def _find_diminishing_returns_point(self, speedup_values: List[float], thread_counts: List[int]) -> int:
        """Find the thread count where diminishing returns begin (improvement < 10%)."""
        for i in range(1, len(speedup_values)):
            if i > 0:
                improvement = (speedup_values[i] - speedup_values[i-1]) / speedup_values[i-1]
                if improvement < 0.1:  # Less than 10% improvement
                    return thread_counts[i-1]
        return thread_counts[-1]  # If no diminishing returns found, return max
    
    def _find_best_strategy_per_thread_count(
        self,
        strategy_analyses: Dict[ParallelizationStrategy, ThreadCountAnalysis],
        thread_counts: List[int]
    ) -> Dict[int, ParallelizationStrategy]:
        """Find the best strategy for each thread count."""
        best_per_thread = {}
        
        for i, thread_count in enumerate(thread_counts):
            best_strategy = None
            best_speedup = 0.0
            
            for strategy, analysis in strategy_analyses.items():
                speedup = analysis.speedup_values[i]
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_strategy = strategy
            
            best_per_thread[thread_count] = best_strategy
        
        return best_per_thread
    
    def _generate_thread_count_recommendations(
        self,
        strategy_analyses: Dict[ParallelizationStrategy, ThreadCountAnalysis]
    ) -> Dict[ParallelizationStrategy, int]:
        """Generate optimal thread count recommendations for each strategy."""
        recommendations = {}
        
        for strategy, analysis in strategy_analyses.items():
            # Recommend the thread count that provides best speedup before diminishing returns
            if analysis.diminishing_returns_point <= analysis.optimal_thread_count:
                recommendations[strategy] = analysis.diminishing_returns_point
            else:
                recommendations[strategy] = analysis.optimal_thread_count
        
        return recommendations


class ParallelizationSimulator:
    """
    Main simulator for testing different parallelization strategies.
    """
    
    def __init__(self):
        """Initialize the parallelization simulator."""
        self.logger = logging.getLogger(__name__)
        self.thread_analyzer = ThreadCountPerformanceAnalyzer(self)
    
    def simulate_strategy(
        self,
        strategy: ParallelizationStrategy,
        thread_count: int,
        transactions: List[TransactionData],
        dependencies: List[StateDependency],
        block_number: int
    ) -> ParallelizationResult:
        """
        Simulate a specific parallelization strategy.
        
        Args:
            strategy: The parallelization strategy to use
            thread_count: Number of threads to simulate
            transactions: List of transactions in the block
            dependencies: List of dependencies between transactions
            block_number: Block number being analyzed
            
        Returns:
            Complete simulation result with performance metrics
        """
        self.logger.info(f"Simulating {strategy.value} with {thread_count} threads on block {block_number}")
        
        # Build dependency graph
        dep_graph = DependencyGraph(transactions, dependencies)
        
        # Run strategy-specific allocation
        if strategy == ParallelizationStrategy.SEQUENTIAL:
            assignments = self._simulate_sequential(transactions)
        elif strategy == ParallelizationStrategy.DEPENDENCY_AWARE:
            assignments = self._simulate_dependency_aware(transactions, thread_count, dep_graph)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Calculate thread metrics
        thread_metrics = self._calculate_thread_metrics(assignments, thread_count)
        
        # Calculate aggregate metrics
        return self._calculate_aggregate_metrics(
            strategy, thread_count, block_number, thread_metrics, transactions
        )
    
    def analyze_thread_count_performance(
        self,
        transactions: List[TransactionData],
        dependencies: List[StateDependency],
        block_number: int,
        thread_counts: Optional[List[int]] = None,
        strategies: Optional[List[ParallelizationStrategy]] = None
    ) -> MultiStrategyAnalysis:
        """
        Convenience method for thread count performance analysis.
        
        This is the main entry point for the research functionality.
        """
        return self.thread_analyzer.analyze_thread_scaling(
            transactions, dependencies, block_number, thread_counts, strategies
        )
    
    def _simulate_sequential(self, transactions: List[TransactionData]) -> List[TransactionAssignment]:
        """Simulate sequential execution (baseline)."""
        assignments = []
        for i, tx in enumerate(transactions):
            assignments.append(TransactionAssignment(
                tx_hash=tx.hash,
                tx_index=tx.transaction_index,
                thread_id=0,
                gas_used=tx.gas_used,
                execution_order=i
            ))
        return assignments
    
    def _simulate_dependency_aware(
        self,
        transactions: List[TransactionData],
        thread_count: int,
        dep_graph: DependencyGraph
    ) -> List[TransactionAssignment]:
        """Simulate dependency-aware batching using topological sorting."""
        # Get dependency chains
        chains = dep_graph.get_dependency_chains()
        independent = dep_graph.get_independent_transactions()
        
        assignments = []
        thread_gas = [0] * thread_count
        assigned_transactions = set()  # Track which transactions have been assigned
        
        # Assign dependency chains to threads (balance by total chain gas)
        for chain in chains:
            chain_gas = sum(dep_graph.transactions[tx_hash].gas_used for tx_hash in chain)
            
            # Find thread with minimum gas
            thread_id = min(range(thread_count), key=lambda i: thread_gas[i])
            thread_gas[thread_id] += chain_gas
            
            # Assign all transactions in chain to this thread
            for order, tx_hash in enumerate(chain):
                if tx_hash not in assigned_transactions:  # Prevent double assignment
                    tx = dep_graph.transactions[tx_hash]
                    assignments.append(TransactionAssignment(
                        tx_hash=tx.hash,
                        tx_index=tx.transaction_index,
                        thread_id=thread_id,
                        gas_used=tx.gas_used,
                        execution_order=order
                    ))
                    assigned_transactions.add(tx_hash)
        
        # Assign independent transactions to balance load (only if not already assigned)
        for tx_hash in independent:
            if tx_hash not in assigned_transactions:  # Prevent double assignment
                tx = dep_graph.transactions[tx_hash]
                thread_id = min(range(thread_count), key=lambda i: thread_gas[i])
                thread_gas[thread_id] += tx.gas_used
                
                assignments.append(TransactionAssignment(
                    tx_hash=tx.hash,
                    tx_index=tx.transaction_index,
                    thread_id=thread_id,
                    gas_used=tx.gas_used,
                    execution_order=0  # Independent transactions can execute immediately
                ))
                assigned_transactions.add(tx_hash)
        
        return assignments
    
    def _calculate_thread_metrics(
        self,
        assignments: List[TransactionAssignment],
        thread_count: int
    ) -> List[ThreadMetrics]:
        """Calculate performance metrics for each thread."""
        threads = defaultdict(list)
        for assignment in assignments:
            threads[assignment.thread_id].append(assignment)
        
        metrics = []
        for thread_id in range(thread_count):
            thread_txs = threads[thread_id]
            total_gas = sum(tx.gas_used for tx in thread_txs)
            
            metrics.append(ThreadMetrics(
                thread_id=thread_id,
                transaction_count=len(thread_txs),
                total_gas=total_gas,
                execution_time_estimate=total_gas / 1000000,  # Rough estimate
                transactions=thread_txs
            ))
        
        return metrics
    
    def _calculate_aggregate_metrics(
        self,
        strategy: ParallelizationStrategy,
        thread_count: int,
        block_number: int,
        thread_metrics: List[ThreadMetrics],
        transactions: List[TransactionData]
    ) -> ParallelizationResult:
        """Calculate aggregate performance metrics."""
        gas_values = [tm.total_gas for tm in thread_metrics if tm.total_gas > 0]
        
        if not gas_values:
            # Edge case: no transactions
            bottleneck_gas = 0
            average_gas = 0.0
            thread_efficiency = 1.0
        else:
            bottleneck_gas = max(gas_values)
            average_gas = sum(gas_values) / len(gas_values)
            thread_efficiency = min(gas_values) / bottleneck_gas if bottleneck_gas > 0 else 1.0
        
        total_gas = sum(tx.gas_used for tx in transactions)
        sequential_gas = total_gas  # All gas on one thread
        parallel_speedup = sequential_gas / bottleneck_gas if bottleneck_gas > 0 else 1.0
        utilization_efficiency = average_gas / bottleneck_gas if bottleneck_gas > 0 else 1.0
        
        return ParallelizationResult(
            strategy=strategy,
            thread_count=thread_count,
            block_number=block_number,
            thread_metrics=thread_metrics,
            bottleneck_gas=bottleneck_gas,
            average_gas=average_gas,
            thread_efficiency=thread_efficiency,
            conflict_rate=0.0,  # TODO: Implement conflict detection
            total_conflicts=0,
            sequential_gas=sequential_gas,
            parallel_speedup=parallel_speedup,
            utilization_efficiency=utilization_efficiency
        ) 