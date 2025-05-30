"""
Transaction fetcher for efficiently retrieving and parsing Ethereum transaction data.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .ethereum_client import EthereumClient


@dataclass
class TransactionData:
    """Data class for storing transaction information."""
    hash: str
    block_number: int
    transaction_index: int
    from_address: str
    to_address: Optional[str]
    value: int
    gas: int
    gas_price: int
    gas_used: Optional[int] = None
    status: Optional[int] = None
    input_data: Optional[str] = None
    logs: Optional[List[Dict]] = None


@dataclass
class BlockData:
    """Data class for storing block information."""
    number: int
    hash: str
    timestamp: int
    gas_used: int
    gas_limit: int
    transaction_count: int
    transactions: List[TransactionData]


class TransactionFetcher:
    """
    Fetches and processes transaction data from Ethereum blocks.
    Handles batch processing and rate limiting.
    """
    
    def __init__(self, client: EthereumClient, max_workers: int = 5):
        """
        Initialize the transaction fetcher.
        
        Args:
            client: Ethereum client instance
            max_workers: Maximum number of concurrent workers for parallel fetching
        """
        self.client = client
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
    
    def fetch_block_with_transactions(self, block_number: int) -> BlockData:
        """
        Fetch a complete block with all transaction data.
        
        Args:
            block_number: Block number to fetch
            
        Returns:
            BlockData object with complete transaction information
        """
        self.logger.info(f"Fetching block {block_number} with full transaction data")
        
        # Get block with transaction hashes
        block = self.client.get_block(block_number, full_transactions=False)
        
        # Extract basic block info
        block_data = BlockData(
            number=block['number'],
            hash=block['hash'].hex() if isinstance(block['hash'], bytes) else block['hash'],
            timestamp=block['timestamp'],
            gas_used=block['gasUsed'],
            gas_limit=block['gasLimit'],
            transaction_count=len(block['transactions']),
            transactions=[]
        )
        
        # Fetch all transactions in parallel
        if block['transactions']:
            self.logger.info(f"Fetching {len(block['transactions'])} transactions from block {block_number}")
            transactions = self._fetch_transactions_parallel(block['transactions'], block_number)
            block_data.transactions = transactions
        
        self.logger.info(f"Successfully fetched block {block_number} with {len(block_data.transactions)} transactions")
        return block_data
    
    def _fetch_transactions_parallel(self, tx_hashes: List[str], block_number: int) -> List[TransactionData]:
        """
        Fetch multiple transactions in parallel.
        
        Args:
            tx_hashes: List of transaction hashes
            block_number: Block number for context
            
        Returns:
            List of TransactionData objects
        """
        transactions = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all transaction fetch tasks
            future_to_hash = {
                executor.submit(self._fetch_single_transaction, tx_hash, block_number, idx): (tx_hash, idx)
                for idx, tx_hash in enumerate(tx_hashes)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_hash):
                tx_hash, tx_index = future_to_hash[future]
                try:
                    tx_data = future.result()
                    if tx_data:
                        transactions.append(tx_data)
                except Exception as e:
                    self.logger.error(f"Failed to fetch transaction {tx_hash}: {e}")
        
        # Sort by transaction index to maintain order
        transactions.sort(key=lambda x: x.transaction_index)
        return transactions
    
    def _fetch_single_transaction(self, tx_hash: str, block_number: int, tx_index: int) -> Optional[TransactionData]:
        """
        Fetch a single transaction with its receipt.
        
        Args:
            tx_hash: Transaction hash
            block_number: Block number
            tx_index: Transaction index in block
            
        Returns:
            TransactionData object or None if failed
        """
        try:
            # Convert bytes hash to hex string if needed
            if isinstance(tx_hash, bytes):
                tx_hash = tx_hash.hex()
            
            # Fetch transaction and receipt
            tx = self.client.get_transaction(tx_hash)
            receipt = self.client.get_transaction_receipt(tx_hash)
            
            # Create TransactionData object
            tx_data = TransactionData(
                hash=tx_hash,
                block_number=block_number,
                transaction_index=tx_index,
                from_address=tx['from'],
                to_address=tx.get('to'),
                value=tx['value'],
                gas=tx['gas'],
                gas_price=tx.get('gasPrice', 0),
                gas_used=receipt.get('gasUsed'),
                status=receipt.get('status'),
                input_data=tx.get('input', '0x'),
                logs=receipt.get('logs', [])
            )
            
            return tx_data
            
        except Exception as e:
            self.logger.error(f"Error fetching transaction {tx_hash}: {e}")
            return None
    
    def fetch_block_range(self, start_block: int, end_block: int) -> List[BlockData]:
        """
        Fetch a range of blocks with transaction data.
        
        Args:
            start_block: Starting block number (inclusive)
            end_block: Ending block number (inclusive)
            
        Returns:
            List of BlockData objects
        """
        self.logger.info(f"Fetching block range {start_block} to {end_block}")
        
        blocks = []
        for block_num in range(start_block, end_block + 1):
            try:
                block_data = self.fetch_block_with_transactions(block_num)
                blocks.append(block_data)
                
                # Add small delay to avoid overwhelming the node
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Failed to fetch block {block_num}: {e}")
                continue
        
        self.logger.info(f"Successfully fetched {len(blocks)} blocks from range {start_block}-{end_block}")
        return blocks
    
    def get_transaction_summary(self, tx_data: TransactionData) -> Dict[str, Any]:
        """
        Get a summary of transaction data for analysis.
        
        Args:
            tx_data: TransactionData object
            
        Returns:
            Dictionary with transaction summary
        """
        return {
            'hash': tx_data.hash,
            'block_number': tx_data.block_number,
            'from': tx_data.from_address,
            'to': tx_data.to_address,
            'value_eth': tx_data.value / 1e18,  # Convert wei to ETH
            'gas_used': tx_data.gas_used,
            'gas_price_gwei': tx_data.gas_price / 1e9 if tx_data.gas_price else 0,  # Convert wei to Gwei
            'status': 'success' if tx_data.status == 1 else 'failed' if tx_data.status == 0 else 'unknown',
            'is_contract_creation': tx_data.to_address is None,
            'has_input_data': tx_data.input_data and tx_data.input_data != '0x',
            'log_count': len(tx_data.logs) if tx_data.logs else 0
        } 