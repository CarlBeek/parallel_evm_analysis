"""
Parallelization simulation framework for analyzing thread allocation performance.
This module simulates how transactions would be distributed across N threads and calculates performance metrics.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple, NamedTuple
from dataclasses import dataclass
from collections import defaultdict
import math
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.transaction_fetcher import TransactionData
from analysis.state_dependency_analyzer import StateDependency


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
    thread_count: int
    block_number: int
    thread_metrics: List[ThreadMetrics]
    
    # Aggregate metrics
    bottleneck_gas: int  # Maximum gas on any thread (determines execution time)
    mean_gas: float  # Mean gas across threads
    thread_efficiency: float  # Ratio of min/max gas (1.0 = perfect balance)
    conflict_rate: float  # Percentage of transactions with conflicts
    total_conflicts: int
    
    # Performance improvement metrics
    sequential_gas: int  # Total gas if executed sequentially (n=1)
    parallel_speedup: float  # Sequential gas / bottleneck gas
    utilization_efficiency: float  # Mean gas / bottleneck gas


@dataclass
class ThreadCountAnalysis:
    """Analysis results across multiple thread counts."""
    block_number: int
    thread_counts: List[int]
    bottleneck_gas_values: List[int]  # Max gas per thread count
    mean_gas_values: List[float]  # Mean gas per thread count
    speedup_values: List[float]  # Speedup vs sequential per thread count
    efficiency_values: List[float]  # Thread efficiency per thread count
    
    # Analysis metrics
    optimal_thread_count: int  # Thread count with best speedup
    diminishing_returns_point: int  # Where improvement drops below threshold
    max_speedup: float  # Best speedup achieved
    efficiency_at_optimal: float  # Thread efficiency at optimal point


@dataclass
class MultiBlockAnalysis:
    """Analysis results across multiple blocks for aggregate statistics."""
    thread_counts: List[int]
    block_numbers: List[int]
    
    # Aggregate statistics per thread count
    mean_bottleneck_gas: List[float]  # Mean max gas across blocks
    ci_lower_bottleneck: List[float]  # 95% CI lower bound
    ci_upper_bottleneck: List[float]  # 95% CI upper bound
    max_bottleneck_gas: List[int]     # Maximum observed across blocks
    
    mean_speedup: List[float]         # Mean speedup across blocks
    ci_lower_speedup: List[float]     # 95% CI lower bound for speedup
    ci_upper_speedup: List[float]     # 95% CI upper bound for speedup
    max_speedup: List[float]          # Maximum speedup observed


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
    Analyzer for studying how thread count affects parallelization performance.
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
        thread_counts: Optional[List[int]] = None
    ) -> ThreadCountAnalysis:
        """
        Analyze how different thread counts affect parallelization performance.
        
        Args:
            transactions: List of transactions to analyze
            dependencies: List of dependencies between transactions
            block_number: Block number being analyzed
            thread_counts: List of thread counts to test (default: [1,2,4,8,16,32,64])
            
        Returns:
            Comprehensive analysis across all thread counts
        """
        if thread_counts is None:
            thread_counts = [1, 2, 4, 8, 16, 32, 64]
        
        self.logger.info(f"Analyzing thread scaling for block {block_number}: "
                        f"{len(thread_counts)} thread counts")
        
        bottleneck_values = []
        mean_values = []
        speedup_values = []
        efficiency_values = []
        
        # Get sequential baseline (n=1) for speedup calculation
        sequential_result = self.simulator.simulate(
            1, transactions, dependencies, block_number
        )
        sequential_gas = sequential_result.sequential_gas
        
        # Test each thread count
        for thread_count in thread_counts:
            result = self.simulator.simulate(
                thread_count, transactions, dependencies, block_number
            )
            
            bottleneck_values.append(result.bottleneck_gas)
            mean_values.append(result.mean_gas)
            speedup_values.append(sequential_gas / result.bottleneck_gas if result.bottleneck_gas > 0 else 1.0)
            efficiency_values.append(result.thread_efficiency)
        
        # Find optimal thread count and diminishing returns point
        optimal_idx = speedup_values.index(max(speedup_values))
        optimal_thread_count = thread_counts[optimal_idx]
        max_speedup = speedup_values[optimal_idx]
        
        # Find diminishing returns point (where improvement drops below 10%)
        diminishing_point = self._find_diminishing_returns_point(speedup_values, thread_counts)
        
        return ThreadCountAnalysis(
            block_number=block_number,
            thread_counts=thread_counts,
            bottleneck_gas_values=bottleneck_values,
            mean_gas_values=mean_values,
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


class ParallelizationSimulator:
    """
    Main simulator for testing parallelization performance across different thread counts.
    """
    
    def __init__(self):
        """Initialize the parallelization simulator."""
        self.logger = logging.getLogger(__name__)
        self.thread_analyzer = ThreadCountPerformanceAnalyzer(self)
    
    def simulate(
        self,
        thread_count: int,
        transactions: List[TransactionData],
        dependencies: List[StateDependency],
        block_number: int
    ) -> ParallelizationResult:
        """
        Simulate parallelization with the given thread count.
        
        Args:
            thread_count: Number of threads to simulate
            transactions: List of transactions in the block
            dependencies: List of dependencies between transactions
            block_number: Block number being analyzed
            
        Returns:
            Complete simulation result with performance metrics
        """
        self.logger.info(f"Simulating parallelization with {thread_count} threads on block {block_number}")
        
        # Build dependency graph
        dep_graph = DependencyGraph(transactions, dependencies)
        
        # Run parallelization allocation
        assignments = self._allocate_transactions(transactions, thread_count, dep_graph)
        
        # Calculate thread metrics
        thread_metrics = self._calculate_thread_metrics(assignments, thread_count)
        
        # Calculate aggregate metrics
        return self._calculate_aggregate_metrics(
            thread_count, block_number, thread_metrics, transactions
        )
    
    def analyze_thread_count_performance(
        self,
        transactions: List[TransactionData],
        dependencies: List[StateDependency],
        block_number: int,
        thread_counts: Optional[List[int]] = None
    ) -> ThreadCountAnalysis:
        """
        Convenience method for thread count performance analysis.
        
        This is the main entry point for the research functionality.
        """
        return self.thread_analyzer.analyze_thread_scaling(
            transactions, dependencies, block_number, thread_counts
        )
    
    def _allocate_transactions(
        self,
        transactions: List[TransactionData],
        thread_count: int,
        dep_graph: DependencyGraph
    ) -> List[TransactionAssignment]:
        """Allocate transactions to threads using dependency-aware approach."""
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
            mean_gas = 0.0
            thread_efficiency = 1.0
        else:
            bottleneck_gas = max(gas_values)
            mean_gas = sum(gas_values) / len(gas_values)
            thread_efficiency = min(gas_values) / bottleneck_gas if bottleneck_gas > 0 else 1.0
        
        total_gas = sum(tx.gas_used for tx in transactions)
        sequential_gas = total_gas  # All gas on one thread
        parallel_speedup = sequential_gas / bottleneck_gas if bottleneck_gas > 0 else 1.0
        utilization_efficiency = mean_gas / bottleneck_gas if bottleneck_gas > 0 else 1.0
        
        return ParallelizationResult(
            thread_count=thread_count,
            block_number=block_number,
            thread_metrics=thread_metrics,
            bottleneck_gas=bottleneck_gas,
            mean_gas=mean_gas,
            thread_efficiency=thread_efficiency,
            conflict_rate=0.0,  # TODO: Implement conflict detection
            total_conflicts=0,
            sequential_gas=sequential_gas,
            parallel_speedup=parallel_speedup,
            utilization_efficiency=utilization_efficiency
        ) 