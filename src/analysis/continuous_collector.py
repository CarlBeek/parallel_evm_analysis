"""
Continuous collector for Ethereum transaction dependency analysis.
Runs continuously to collect and analyze recent blocks within the debug trace availability window.
"""

import logging
import time
import signal
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ethereum_client import EthereumClient
from core.transaction_fetcher import TransactionFetcher, BlockData
from storage.database import BlockchainDatabase
from analysis.state_dependency_analyzer import StateDependencyAnalyzer


@dataclass
class CollectionStats:
    """Statistics for continuous collection."""
    blocks_analyzed: int = 0
    transactions_analyzed: int = 0
    dependencies_found: int = 0
    errors_encountered: int = 0
    start_time: Optional[datetime] = None
    last_block_analyzed: Optional[int] = None
    last_analysis_time: Optional[datetime] = None


class ContinuousCollector:
    """
    Continuous collector for Ethereum transaction dependency analysis.
    Monitors for new blocks and analyzes them while debug tracing is available.
    """
    
    def __init__(self, 
                 client: EthereumClient,
                 database: BlockchainDatabase,
                 lookback_blocks: int = 100,
                 check_interval: int = 60,
                 max_workers: int = 4):
        """
        Initialize the continuous collector.
        
        Args:
            client: EthereumClient for blockchain access
            database: Database for storing results
            lookback_blocks: How many blocks back from latest to consider (max 128 for path scheme)
            check_interval: How often to check for new blocks (seconds)
            max_workers: Number of parallel workers for analysis
        """
        self.client = client
        self.database = database
        self.lookback_blocks = min(lookback_blocks, 100)  # Stay safe within 128 limit
        self.check_interval = check_interval
        self.max_workers = max_workers
        
        self.logger = logging.getLogger(__name__)
        self.stats = CollectionStats()
        self.running = False
        self.fetcher = TransactionFetcher(client)
        
        # Initialize analyzer if debug API is available
        try:
            self.analyzer = StateDependencyAnalyzer(client, database, max_workers)
            self.debug_available = True
            self.logger.info("Debug API available - full state dependency analysis enabled")
        except RuntimeError as e:
            self.logger.warning(f"Debug API not available: {e}")
            self.debug_available = False
            self.analyzer = None
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info(f"Continuous collector initialized:")
        self.logger.info(f"  Lookback blocks: {self.lookback_blocks}")
        self.logger.info(f"  Check interval: {self.check_interval}s")
        self.logger.info(f"  Max workers: {self.max_workers}")
        self.logger.info(f"  Debug analysis: {'enabled' if self.debug_available else 'disabled'}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
    
    def start(self):
        """Start continuous collection."""
        self.logger.info("Starting continuous collection...")
        self.running = True
        self.stats.start_time = datetime.now()
        
        try:
            self._collection_loop()
        except Exception as e:
            self.logger.error(f"Fatal error in collection loop: {e}")
            raise
        finally:
            self._log_final_stats()
    
    def stop(self):
        """Stop continuous collection."""
        self.logger.info("Stopping continuous collection...")
        self.running = False
    
    def _collection_loop(self):
        """Main collection loop."""
        self.logger.info("Collection loop started")
        
        while self.running:
            try:
                # Get current blockchain state
                latest_block = self.client.get_latest_block_number()
                
                if latest_block is None:
                    self.logger.warning("Could not get latest block number, retrying...")
                    time.sleep(self.check_interval)
                    continue
                
                # Determine block range to analyze
                start_block = max(latest_block - self.lookback_blocks, 0)
                end_block = latest_block - 5  # Leave 5 blocks buffer for finalization
                
                if start_block > end_block:
                    self.logger.debug(f"No blocks to analyze yet (latest: {latest_block})")
                    time.sleep(self.check_interval)
                    continue
                
                # Find blocks that need analysis
                blocks_to_analyze = self._find_blocks_to_analyze(start_block, end_block)
                
                if not blocks_to_analyze:
                    self.logger.debug(f"No new blocks to analyze (range: {start_block}-{end_block})")
                    time.sleep(self.check_interval)
                    continue
                
                self.logger.info(f"Analyzing {len(blocks_to_analyze)} new blocks: {blocks_to_analyze[0]}-{blocks_to_analyze[-1]}")
                
                # Analyze blocks
                for block_num in blocks_to_analyze:
                    if not self.running:
                        break
                    
                    try:
                        self._analyze_block(block_num)
                        self.stats.last_block_analyzed = block_num
                        self.stats.last_analysis_time = datetime.now()
                    except Exception as e:
                        self.logger.error(f"Error analyzing block {block_num}: {e}")
                        self.stats.errors_encountered += 1
                        continue
                
                # Log progress
                if blocks_to_analyze:
                    self._log_progress()
                
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                self.stats.errors_encountered += 1
                time.sleep(self.check_interval)
                continue
            
            # Wait before next check
            if self.running:
                time.sleep(self.check_interval)
    
    def _find_blocks_to_analyze(self, start_block: int, end_block: int) -> List[int]:
        """Find blocks in range that haven't been analyzed yet."""
        blocks_to_analyze = []
        
        for block_num in range(start_block, end_block + 1):
            # Check if block has been analyzed
            if not self.database.has_block_analysis(block_num):
                blocks_to_analyze.append(block_num)
        
        return blocks_to_analyze
    
    def _analyze_block(self, block_num: int):
        """Analyze a single block."""
        start_time = time.time()
        
        try:
            # Fetch block data
            self.logger.debug(f"Fetching block {block_num}...")
            block_data = self.fetcher.fetch_block_with_transactions(block_num)
            
            if not block_data or not block_data.transactions:
                self.logger.warning(f"Block {block_num} has no transactions, skipping")
                return
            
            # Store basic block data
            self.database.store_block(block_data)
            
            # Perform dependency analysis if debug API is available
            dependencies = []
            if self.debug_available and self.analyzer:
                try:
                    dependencies = self.analyzer.analyze_block_state_dependencies(block_data)
                    self.logger.info(f"Block {block_num}: found {len(dependencies)} state dependencies")
                except Exception as e:
                    self.logger.warning(f"State dependency analysis failed for block {block_num}: {e}")
            
            # Update statistics
            self.stats.blocks_analyzed += 1
            self.stats.transactions_analyzed += len(block_data.transactions)
            self.stats.dependencies_found += len(dependencies)
            
            analysis_time = time.time() - start_time
            self.logger.info(f"Block {block_num} analyzed in {analysis_time:.1f}s: "
                           f"{len(block_data.transactions)} txs, {len(dependencies)} deps")
            
        except Exception as e:
            self.logger.error(f"Failed to analyze block {block_num}: {e}")
            raise
    
    def _log_progress(self):
        """Log collection progress."""
        if not self.stats.start_time:
            return
        
        runtime = datetime.now() - self.stats.start_time
        
        self.logger.info("Collection Progress:")
        self.logger.info(f"  Runtime: {runtime}")
        self.logger.info(f"  Blocks analyzed: {self.stats.blocks_analyzed}")
        self.logger.info(f"  Transactions analyzed: {self.stats.transactions_analyzed}")
        self.logger.info(f"  Dependencies found: {self.stats.dependencies_found}")
        self.logger.info(f"  Errors encountered: {self.stats.errors_encountered}")
        if self.stats.last_block_analyzed:
            self.logger.info(f"  Last block analyzed: {self.stats.last_block_analyzed}")
    
    def _log_final_stats(self):
        """Log final collection statistics."""
        if not self.stats.start_time:
            return
        
        runtime = datetime.now() - self.stats.start_time
        
        self.logger.info("Final Collection Statistics:")
        self.logger.info(f"  Total runtime: {runtime}")
        self.logger.info(f"  Blocks analyzed: {self.stats.blocks_analyzed}")
        self.logger.info(f"  Transactions analyzed: {self.stats.transactions_analyzed}")
        self.logger.info(f"  Dependencies found: {self.stats.dependencies_found}")
        self.logger.info(f"  Errors encountered: {self.stats.errors_encountered}")
        
        if self.stats.blocks_analyzed > 0:
            avg_txs_per_block = self.stats.transactions_analyzed / self.stats.blocks_analyzed
            avg_deps_per_block = self.stats.dependencies_found / self.stats.blocks_analyzed
            self.logger.info(f"  Mean transactions per block: {avg_txs_per_block:.1f}")
            self.logger.info(f"  Mean dependencies per block: {avg_deps_per_block:.1f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current collection statistics."""
        runtime = None
        if self.stats.start_time:
            runtime = datetime.now() - self.stats.start_time
        
        return {
            'running': self.running,
            'start_time': self.stats.start_time,
            'runtime': runtime,
            'blocks_analyzed': self.stats.blocks_analyzed,
            'transactions_analyzed': self.stats.transactions_analyzed,
            'dependencies_found': self.stats.dependencies_found,
            'errors_encountered': self.stats.errors_encountered,
            'last_block_analyzed': self.stats.last_block_analyzed,
            'last_analysis_time': self.stats.last_analysis_time,
            'debug_available': self.debug_available
        }
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get a summary of what has been collected."""
        # Get database statistics
        db_stats = self.database.get_database_stats()
        
        return {
            'collection_stats': self.get_stats(),
            'database_stats': db_stats,
            'configuration': {
                'lookback_blocks': self.lookback_blocks,
                'check_interval': self.check_interval,
                'max_workers': self.max_workers,
                'debug_available': self.debug_available
            }
        } 