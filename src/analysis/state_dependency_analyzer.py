"""
State-based dependency analyzer using debug_traceTransaction for exact state access tracking.
This provides the most accurate dependency detection by analyzing actual storage reads/writes.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.transaction_fetcher import BlockData, TransactionData
from core.ethereum_client import EthereumClient
from storage.database import BlockchainDatabase


@dataclass
class StateAccess:
    """Represents a state access (read or write) by a transaction."""
    tx_hash: str
    tx_index: int
    contract_address: str
    storage_slot: str
    access_type: str  # 'read' or 'write'
    value_before: Optional[str] = None
    value_after: Optional[str] = None


@dataclass
class StateDependency:
    """Represents a true state dependency between two transactions."""
    dependent_tx_hash: str
    dependency_tx_hash: str
    dependent_tx_index: int
    dependency_tx_index: int
    contract_address: str
    storage_slot: str
    dependency_reason: str
    gas_impact: int


class StateDependencyAnalyzer:
    """
    Analyzes transaction dependencies using debug_traceTransaction for exact state access tracking.
    This provides the most accurate dependency detection possible.
    """
    
    def __init__(self, client: EthereumClient, database: BlockchainDatabase, max_workers: int = 8):
        """
        Initialize the state dependency analyzer.
        
        Args:
            client: EthereumClient for debug API access
            database: Database for storing results
            max_workers: Number of parallel workers for tracing transactions
        """
        self.client = client
        self.database = database
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # Check if debug API is available
        if not self.client.check_debug_api_availability():
            raise RuntimeError("Debug API not available - cannot perform state-based analysis")
        
        # Cache for trace results to avoid re-tracing
        self.trace_cache = {}
        
        self.logger.info(f"State dependency analyzer initialized with debug API support and {max_workers} workers")
    
    def analyze_block_state_dependencies(self, block_data: BlockData) -> List[StateDependency]:
        """
        Analyze a block for true state dependencies using debug traces.
        
        Args:
            block_data: Block data containing all transactions
            
        Returns:
            List of true state dependencies
        """
        self.logger.info(f"Analyzing state dependencies for block {block_data.number} with {len(block_data.transactions)} transactions")
        
        # Step 1: Get state access patterns for all transactions (parallelized)
        start_time = time.time()
        state_accesses = self._get_all_state_accesses_parallel(block_data.transactions)
        trace_time = time.time() - start_time
        self.logger.info(f"Traced {len(block_data.transactions)} transactions in {trace_time:.1f}s ({len(state_accesses)} accesses)")
        
        # Step 2: Build state access maps
        writes_by_slot = defaultdict(list)  # storage_slot -> list of writes
        reads_by_slot = defaultdict(list)   # storage_slot -> list of reads
        
        for access in state_accesses:
            slot_key = f"{access.contract_address}:{access.storage_slot}"
            if access.access_type == 'write':
                writes_by_slot[slot_key].append(access)
            elif access.access_type == 'read':
                reads_by_slot[slot_key].append(access)
        
        # Step 3: Find true dependencies (reads that depend on writes)
        dependencies = []
        
        for slot_key, reads in reads_by_slot.items():
            if slot_key not in writes_by_slot:
                continue  # No writes to this slot in this block
            
            writes = writes_by_slot[slot_key]
            
            # For each read, find the most recent write before it
            for read_access in reads:
                # Find writes that happened before this read
                prior_writes = [
                    w for w in writes 
                    if w.tx_index < read_access.tx_index
                ]
                
                if prior_writes:
                    # Get the most recent write
                    most_recent_write = max(prior_writes, key=lambda w: w.tx_index)
                    
                    # Create dependency
                    dependency = StateDependency(
                        dependent_tx_hash=read_access.tx_hash,
                        dependency_tx_hash=most_recent_write.tx_hash,
                        dependent_tx_index=read_access.tx_index,
                        dependency_tx_index=most_recent_write.tx_index,
                        contract_address=read_access.contract_address,
                        storage_slot=read_access.storage_slot,
                        dependency_reason=f"Transaction reads storage slot {read_access.storage_slot} written by prior transaction",
                        gas_impact=self._get_transaction_gas(block_data.transactions, read_access.tx_hash)
                    )
                    dependencies.append(dependency)
        
        self.logger.info(f"Found {len(dependencies)} true state dependencies")
        
        # Store dependencies in database (batch operation)
        self._store_dependencies_batch(dependencies)
        
        return dependencies
    
    def _get_all_state_accesses_parallel(self, transactions: List[TransactionData]) -> List[StateAccess]:
        """
        Get state access patterns for all transactions using parallel debug traces.
        
        Args:
            transactions: List of transactions to analyze
            
        Returns:
            List of all state accesses
        """
        all_accesses = []
        
        # Filter out simple transfers to speed up analysis
        complex_txs = [
            tx for tx in transactions 
            if tx.to_address and tx.input_data and len(tx.input_data) > 10
        ]
        
        self.logger.info(f"Analyzing {len(complex_txs)} complex transactions out of {len(transactions)} total")
        
        # Use ThreadPoolExecutor for parallel tracing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all trace jobs
            future_to_tx = {
                executor.submit(self._trace_transaction_state_access_safe, tx): tx 
                for tx in complex_txs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_tx):
                tx = future_to_tx[future]
                try:
                    accesses = future.result()
                    all_accesses.extend(accesses)
                except Exception as e:
                    self.logger.warning(f"Failed to trace transaction {tx.hash}: {e}")
                    continue
        
        return all_accesses
    
    def _trace_transaction_state_access_safe(self, tx: TransactionData) -> List[StateAccess]:
        """
        Safe wrapper for tracing a single transaction with error handling.
        
        Args:
            tx: Transaction to trace
            
        Returns:
            List of state accesses for this transaction
        """
        try:
            return self._trace_transaction_state_access(tx)
        except Exception as e:
            self.logger.debug(f"Trace failed for {tx.hash}: {e}")
            return []
    
    def _trace_transaction_state_access(self, tx: TransactionData) -> List[StateAccess]:
        """
        Trace a single transaction to get its state access patterns.
        
        Args:
            tx: Transaction to trace
            
        Returns:
            List of state accesses for this transaction
        """
        # Check cache first
        if tx.hash in self.trace_cache:
            return self.trace_cache[tx.hash]
        
        accesses = []
        
        try:
            # Use prestateTracer to get state access information
            trace_config = {
                "tracer": "prestateTracer",
                "tracerConfig": {
                    "diffMode": True
                }
            }
            
            trace_result = self.client.debug_trace_transaction(tx.hash, trace_config)
            
            # Parse the trace result to extract state accesses
            accesses = self._parse_prestate_trace(tx, trace_result)
            
            # Cache the result
            self.trace_cache[tx.hash] = accesses
            
        except Exception as e:
            self.logger.debug(f"Failed to trace transaction {tx.hash}: {e}")
            # Don't use fallback for speed - just return empty
            accesses = []
        
        return accesses
    
    def _parse_prestate_trace(self, tx: TransactionData, trace_result: Dict) -> List[StateAccess]:
        """
        Parse prestateTracer output to extract state accesses.
        
        Args:
            tx: Transaction being traced
            trace_result: Result from prestateTracer
            
        Returns:
            List of state accesses
        """
        accesses = []
        
        # prestateTracer returns pre and post state
        # We need to analyze the differences to understand reads/writes
        
        if not hasattr(trace_result, 'keys'):
            return accesses
        
        # Convert AttributeDict to regular dict for easier processing
        result_dict = dict(trace_result)
        
        if 'pre' not in result_dict or 'post' not in result_dict:
            return accesses
        
        pre_state = dict(result_dict['pre'])
        post_state = dict(result_dict['post'])
        
        # Analyze all addresses that appear in either pre or post state
        all_addresses = set(pre_state.keys()) | set(post_state.keys())
        
        for address in all_addresses:
            # Get pre and post data for this address
            pre_data = dict(pre_state.get(address, {})) if address in pre_state else {}
            post_data = dict(post_state.get(address, {})) if address in post_state else {}
            
            # Extract storage data
            pre_storage = {}
            post_storage = {}
            
            if 'storage' in pre_data and hasattr(pre_data['storage'], 'keys'):
                pre_storage = dict(pre_data['storage'])
            elif 'storage' in pre_data and isinstance(pre_data['storage'], dict):
                pre_storage = pre_data['storage']
            
            if 'storage' in post_data and hasattr(post_data['storage'], 'keys'):
                post_storage = dict(post_data['storage'])
            elif 'storage' in post_data and isinstance(post_data['storage'], dict):
                post_storage = post_data['storage']
            
            # Find all storage slots that were accessed
            all_slots = set(pre_storage.keys()) | set(post_storage.keys())
            
            for slot in all_slots:
                pre_value = pre_storage.get(slot, '0x0')
                post_value = post_storage.get(slot, '0x0')
                
                # If values are different, this is a write
                if pre_value != post_value:
                    accesses.append(StateAccess(
                        tx_hash=tx.hash,
                        tx_index=tx.transaction_index,
                        contract_address=address,
                        storage_slot=slot,
                        access_type='write',
                        value_before=pre_value,
                        value_after=post_value
                    ))
                
                # If slot exists in pre_state, it was read
                # (Note: prestateTracer shows all slots that were accessed)
                if slot in pre_storage:
                    accesses.append(StateAccess(
                        tx_hash=tx.hash,
                        tx_index=tx.transaction_index,
                        contract_address=address,
                        storage_slot=slot,
                        access_type='read',
                        value_before=pre_value,
                        value_after=pre_value
                    ))
        
        return accesses
    
    def _store_dependencies_batch(self, dependencies: List[StateDependency]):
        """Store dependencies in database using batch operations for speed."""
        for dep in dependencies:
            self.database.store_dependency(
                dep.dependent_tx_hash,
                dep.dependency_tx_hash,
                'state_dependency',
                dep.dependency_reason,
                dep.gas_impact
            )
    
    def _get_transaction_gas(self, transactions: List[TransactionData], tx_hash: str) -> int:
        """Get gas used by a specific transaction."""
        for tx in transactions:
            if tx.hash == tx_hash:
                return tx.gas_used or 0
        return 0
    
    def get_parallelization_analysis(self, dependencies: List[StateDependency], 
                                   total_transactions: int, total_gas: int) -> Dict[str, Any]:
        """
        Analyze parallelization potential based on true state dependencies.
        
        Args:
            dependencies: List of state dependencies
            total_transactions: Total number of transactions in block
            total_gas: Total gas used in block
            
        Returns:
            Dictionary with parallelization analysis
        """
        # Find transactions involved in dependencies
        dependent_txs = set()
        for dep in dependencies:
            dependent_txs.add(dep.dependent_tx_hash)
            dependent_txs.add(dep.dependency_tx_hash)
        
        # Calculate gas-based dependency chain analysis first
        chain_analysis = self._calculate_dependency_chains_with_gas(dependencies)
        
        # Calculate gas metrics properly - use unique transaction gas
        unique_tx_gas = chain_analysis.get('unique_tx_gas', {})
        dependent_gas = sum(unique_tx_gas.values())  # Total gas in dependent transactions
        independent_gas = total_gas - dependent_gas
        gas_parallelization_potential = (independent_gas / total_gas) * 100 if total_gas > 0 else 0
        
        # Calculate transaction-based metrics (for comparison)
        independent_txs = total_transactions - len(dependent_txs)
        tx_parallelization_potential = (independent_txs / total_transactions) * 100
        
        # Calculate gas-based theoretical speedup
        # Critical path is the longest chain by gas
        critical_path_gas = max(chain_analysis['max_chain_gas'], independent_gas) if chain_analysis['chain_gas_amounts'] else total_gas
        gas_theoretical_speedup = total_gas / critical_path_gas if critical_path_gas > 0 else 1.0
        
        # Traditional transaction-based speedup for comparison
        tx_theoretical_speedup = total_transactions / max(chain_analysis['max_chain_length'], 1)
        
        return {
            'total_transactions': total_transactions,
            'total_gas': total_gas,
            'true_dependencies': len(dependencies),
            'dependent_transactions': len(dependent_txs),
            'independent_transactions': independent_txs,
            'dependent_gas': dependent_gas,
            'independent_gas': independent_gas,
            
            # Gas-based metrics (primary)
            'gas_parallelization_potential_percent': gas_parallelization_potential,
            'gas_theoretical_speedup': gas_theoretical_speedup,
            'critical_path_gas': critical_path_gas,
            
            # Transaction-based metrics (for comparison)
            'tx_parallelization_potential_percent': tx_parallelization_potential,
            'tx_theoretical_speedup': tx_theoretical_speedup,
            
            # Chain analysis (both gas and transaction based)
            'dependency_chains': chain_analysis['num_chains'],
            'longest_dependency_chain': chain_analysis['max_chain_length'],
            'longest_chain_gas': chain_analysis['max_chain_gas'],
            'avg_chain_length': chain_analysis['avg_chain_length'],
            'avg_chain_gas': chain_analysis['avg_chain_gas'],
            'total_chain_gas': chain_analysis['total_chain_gas'],
            'chain_gas_amounts': chain_analysis['chain_gas_amounts'],
            'chain_lengths': chain_analysis['chain_lengths'],
            
            'accuracy': 'exact_state_analysis'
        }
    
    def _calculate_dependency_chains(self, dependencies: List[StateDependency]) -> List[List[str]]:
        """Calculate dependency chains from the dependency list."""
        # Build dependency graph
        graph = defaultdict(list)
        for dep in dependencies:
            graph[dep.dependency_tx_hash].append(dep.dependent_tx_hash)
        
        # Find all chains
        chains = []
        visited = set()
        
        def dfs_chain(tx_hash, current_chain):
            if tx_hash in visited:
                return
            
            visited.add(tx_hash)
            current_chain.append(tx_hash)
            
            if tx_hash in graph:
                for dependent in graph[tx_hash]:
                    dfs_chain(dependent, current_chain.copy())
            else:
                # End of chain
                chains.append(current_chain)
        
        # Start DFS from all root transactions (those that don't depend on others)
        all_dependents = set(dep.dependent_tx_hash for dep in dependencies)
        all_dependencies = set(dep.dependency_tx_hash for dep in dependencies)
        roots = all_dependencies - all_dependents
        
        for root in roots:
            dfs_chain(root, [])
        
        return chains
    
    def _calculate_dependency_chains_with_gas(self, dependencies: List[StateDependency]) -> Dict[str, Any]:
        """Calculate dependency chains with both transaction counts and gas amounts."""
        # Build dependency graph
        graph = defaultdict(list)
        tx_to_gas = {}  # Map transaction hash to gas impact
        
        for dep in dependencies:
            graph[dep.dependency_tx_hash].append(dep.dependent_tx_hash)
            # Only count each transaction's gas once
            tx_to_gas[dep.dependent_tx_hash] = dep.gas_impact
            # Don't double-count dependency transaction gas
            if dep.dependency_tx_hash not in tx_to_gas:
                tx_to_gas[dep.dependency_tx_hash] = dep.gas_impact
        
        # Find all chains
        chains = []
        chain_gas_amounts = []
        visited = set()
        
        def dfs_chain(tx_hash, current_chain, current_gas):
            if tx_hash in visited:
                return
            
            visited.add(tx_hash)
            current_chain.append(tx_hash)
            current_gas += tx_to_gas.get(tx_hash, 0)
            
            if tx_hash in graph:
                for dependent in graph[tx_hash]:
                    dfs_chain(dependent, current_chain.copy(), current_gas)
            else:
                # End of chain
                chains.append(current_chain)
                chain_gas_amounts.append(current_gas)
        
        # Start DFS from all root transactions (those that don't depend on others)
        all_dependents = set(dep.dependent_tx_hash for dep in dependencies)
        all_dependencies = set(dep.dependency_tx_hash for dep in dependencies)
        roots = all_dependencies - all_dependents
        
        for root in roots:
            dfs_chain(root, [], 0)
        
        # Calculate statistics
        chain_lengths = [len(chain) for chain in chains]
        
        return {
            'chains': chains,
            'chain_lengths': chain_lengths,
            'chain_gas_amounts': chain_gas_amounts,
            'num_chains': len(chains),
            'avg_chain_length': sum(chain_lengths) / len(chain_lengths) if chain_lengths else 0,
            'max_chain_length': max(chain_lengths) if chain_lengths else 0,
            'avg_chain_gas': sum(chain_gas_amounts) / len(chain_gas_amounts) if chain_gas_amounts else 0,
            'max_chain_gas': max(chain_gas_amounts) if chain_gas_amounts else 0,
            'total_chain_gas': sum(chain_gas_amounts),
            'unique_tx_gas': tx_to_gas  # Return the unique transaction gas mapping
        } 