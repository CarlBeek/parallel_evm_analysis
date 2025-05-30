"""
Ethereum client for connecting to and interacting with Ethereum nodes.
"""

import logging
from typing import Dict, Any, Optional, List
from web3 import Web3
from web3.middleware import geth_poa_middleware
import yaml
from pathlib import Path


class EthereumClient:
    """
    Client for interacting with Ethereum nodes (Geth or Reth).
    Handles connection, basic queries, and debug API calls.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Ethereum client.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.yaml
        """
        self.logger = logging.getLogger(__name__)
        self.w3: Optional[Web3] = None
        self.config = self._load_config(config_path)
        self._setup_connection()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default to config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            # Return default config
            return {
                'ethereum': {
                    'node_type': 'geth',
                    'rpc_url': 'http://10.0.30.105:8545',
                    'ws_url': 'ws://10.0.30.105:8546',
                    'required_modules': ['eth', 'net', 'web3', 'debug'],
                    'timeout': 30
                }
            }
    
    def _setup_connection(self):
        """Establish connection to the Ethereum node."""
        eth_config = self.config.get('ethereum', {})
        rpc_url = eth_config.get('rpc_url', 'http://10.0.30.105:8545')
        
        try:
            self.logger.info(f"Connecting to Ethereum node at {rpc_url}")
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            
            # Add PoA middleware if needed (common for testnets)
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Test connection
            if self.w3.is_connected():
                self.logger.info("Successfully connected to Ethereum node")
                self._log_node_info()
            else:
                raise ConnectionError("Failed to connect to Ethereum node")
                
        except Exception as e:
            self.logger.error(f"Failed to connect to Ethereum node: {e}")
            raise
    
    def _log_node_info(self):
        """Log basic information about the connected node."""
        try:
            client_version = self.w3.client_version
            chain_id = self.w3.eth.chain_id
            latest_block = self.w3.eth.block_number
            
            self.logger.info(f"Node client: {client_version}")
            self.logger.info(f"Chain ID: {chain_id}")
            self.logger.info(f"Latest block: {latest_block}")
            
        except Exception as e:
            self.logger.warning(f"Could not retrieve node info: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to the Ethereum node."""
        return self.w3 is not None and self.w3.is_connected()
    
    def get_chain_id(self) -> int:
        """Get the chain ID."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Ethereum node")
        return self.w3.eth.chain_id
    
    def get_latest_block_number(self) -> int:
        """Get the latest block number."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Ethereum node")
        return self.w3.eth.block_number
    
    def get_block(self, block_number: int, full_transactions: bool = True) -> Dict[str, Any]:
        """
        Get block data by block number.
        
        Args:
            block_number: Block number to fetch
            full_transactions: If True, include full transaction objects
            
        Returns:
            Block data dictionary
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Ethereum node")
        
        try:
            block = self.w3.eth.get_block(block_number, full_transactions=full_transactions)
            self.logger.debug(f"Retrieved block {block_number} with {len(block.transactions)} transactions")
            return dict(block)
        except Exception as e:
            self.logger.error(f"Failed to get block {block_number}: {e}")
            raise
    
    def get_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """
        Get transaction data by hash.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction data dictionary
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Ethereum node")
        
        try:
            tx = self.w3.eth.get_transaction(tx_hash)
            return dict(tx)
        except Exception as e:
            self.logger.error(f"Failed to get transaction {tx_hash}: {e}")
            raise
    
    def get_transaction_receipt(self, tx_hash: str) -> Dict[str, Any]:
        """
        Get transaction receipt by hash.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction receipt dictionary
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Ethereum node")
        
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            return dict(receipt)
        except Exception as e:
            self.logger.error(f"Failed to get transaction receipt {tx_hash}: {e}")
            raise
    
    def debug_trace_transaction(self, tx_hash: str, tracer_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get detailed execution trace for a transaction using debug_traceTransaction.
        
        Args:
            tx_hash: Transaction hash
            tracer_config: Configuration for the tracer
            
        Returns:
            Trace data dictionary
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Ethereum node")
        
        # Default tracer config to capture storage access
        if tracer_config is None:
            tracer_config = {
                "tracer": "prestateTracer",
                "tracerConfig": {
                    "diffMode": True
                }
            }
        
        try:
            # Use the debug API
            trace = self.w3.manager.request_blocking("debug_traceTransaction", [tx_hash, tracer_config])
            self.logger.debug(f"Retrieved trace for transaction {tx_hash}")
            return trace
        except Exception as e:
            self.logger.error(f"Failed to trace transaction {tx_hash}: {e}")
            raise
    
    def check_debug_api_availability(self) -> bool:
        """
        Check if debug APIs are available on the connected node.
        
        Returns:
            True if debug APIs are available, False otherwise
        """
        if not self.is_connected():
            return False
        
        try:
            # Get a recent transaction to test with
            latest_block = self.w3.eth.block_number
            block = self.w3.eth.get_block(latest_block - 1, full_transactions=True)
            
            if not block.transactions:
                # Try another block if no transactions
                block = self.w3.eth.get_block(latest_block - 2, full_transactions=True)
            
            if not block.transactions:
                # If still no transactions, assume debug API is not available
                self.logger.warning("No transactions found to test debug API")
                return False
            
            # Use the first transaction hash for testing
            test_tx_hash = block.transactions[0]['hash'].hex()
            
            # Try to call debug_traceTransaction with a simple tracer
            self.w3.manager.request_blocking("debug_traceTransaction", [test_tx_hash, {"tracer": "callTracer"}])
            return True
        except Exception as e:
            self.logger.warning(f"Debug API not available: {e}")
            return False
    
    def get_node_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the connected node.
        
        Returns:
            Dictionary with node information
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Ethereum node")
        
        info = {}
        try:
            info['client_version'] = self.w3.client_version
            info['chain_id'] = self.w3.eth.chain_id
            info['latest_block'] = self.w3.eth.block_number
            info['syncing'] = self.w3.eth.syncing
            info['gas_price'] = self.w3.eth.gas_price
            info['debug_api_available'] = self.check_debug_api_availability()
            
        except Exception as e:
            self.logger.error(f"Error getting node info: {e}")
            
        return info 