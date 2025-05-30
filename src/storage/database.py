"""
SQLite database module for storing blockchain data and analysis results.
"""

import logging
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import asdict
import json
from datetime import datetime
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.transaction_fetcher import BlockData, TransactionData


class BlockchainDatabase:
    """
    SQLite database for storing blockchain data and dependency analysis results.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default from config.
        """
        self.logger = logging.getLogger(__name__)
        
        if db_path is None:
            # Default to data directory
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = data_dir / "ethereum_deps.db"
        
        self.db_path = str(db_path)
        self.connection: Optional[sqlite3.Connection] = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            self.logger.info(f"Connected to SQLite database at {self.db_path}")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            cursor = self.connection.cursor()
            
            # Blocks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS blocks (
                    number INTEGER PRIMARY KEY,
                    hash TEXT NOT NULL UNIQUE,
                    timestamp INTEGER NOT NULL,
                    gas_used INTEGER NOT NULL,
                    gas_limit INTEGER NOT NULL,
                    transaction_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    hash TEXT PRIMARY KEY,
                    block_number INTEGER NOT NULL,
                    transaction_index INTEGER NOT NULL,
                    from_address TEXT NOT NULL,
                    to_address TEXT,
                    value TEXT NOT NULL,
                    gas INTEGER NOT NULL,
                    gas_price TEXT NOT NULL,
                    gas_used INTEGER,
                    status INTEGER,
                    input_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (block_number) REFERENCES blocks (number)
                )
            """)
            
            # Transaction logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transaction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_hash TEXT NOT NULL,
                    log_index INTEGER NOT NULL,
                    address TEXT NOT NULL,
                    topics TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (transaction_hash) REFERENCES transactions (hash),
                    UNIQUE(transaction_hash, log_index)
                )
            """)
            
            # Contract interactions table (for dependency analysis)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS contract_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_hash TEXT NOT NULL,
                    contract_address TEXT NOT NULL,
                    function_selector TEXT,
                    interaction_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (transaction_hash) REFERENCES transactions (hash)
                )
            """)
            
            # Transaction dependencies table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transaction_dependencies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dependent_tx_hash TEXT NOT NULL,
                    dependency_tx_hash TEXT NOT NULL,
                    dependency_type TEXT NOT NULL,
                    dependency_reason TEXT,
                    gas_impact INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dependent_tx_hash) REFERENCES transactions (hash),
                    FOREIGN KEY (dependency_tx_hash) REFERENCES transactions (hash),
                    UNIQUE(dependent_tx_hash, dependency_tx_hash, dependency_type)
                )
            """)
            
            # Analysis results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    block_number INTEGER NOT NULL,
                    analysis_type TEXT NOT NULL,
                    results TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (block_number) REFERENCES blocks (number)
                )
            """)
            
            # Create indexes separately
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_blocks_timestamp ON blocks (timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_blocks_gas_used ON blocks (gas_used)",
                "CREATE INDEX IF NOT EXISTS idx_transactions_block_number ON transactions (block_number)",
                "CREATE INDEX IF NOT EXISTS idx_transactions_from_address ON transactions (from_address)",
                "CREATE INDEX IF NOT EXISTS idx_transactions_to_address ON transactions (to_address)",
                "CREATE INDEX IF NOT EXISTS idx_transactions_gas_used ON transactions (gas_used)",
                "CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions (status)",
                "CREATE INDEX IF NOT EXISTS idx_transaction_logs_hash ON transaction_logs (transaction_hash)",
                "CREATE INDEX IF NOT EXISTS idx_transaction_logs_address ON transaction_logs (address)",
                "CREATE INDEX IF NOT EXISTS idx_contract_interactions_hash ON contract_interactions (transaction_hash)",
                "CREATE INDEX IF NOT EXISTS idx_contract_interactions_address ON contract_interactions (contract_address)",
                "CREATE INDEX IF NOT EXISTS idx_contract_interactions_selector ON contract_interactions (function_selector)",
                "CREATE INDEX IF NOT EXISTS idx_dependencies_dependent ON transaction_dependencies (dependent_tx_hash)",
                "CREATE INDEX IF NOT EXISTS idx_dependencies_dependency ON transaction_dependencies (dependency_tx_hash)",
                "CREATE INDEX IF NOT EXISTS idx_dependencies_type ON transaction_dependencies (dependency_type)",
                "CREATE INDEX IF NOT EXISTS idx_analysis_block_number ON analysis_results (block_number)",
                "CREATE INDEX IF NOT EXISTS idx_analysis_type ON analysis_results (analysis_type)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            self.connection.commit()
            self.logger.info("Database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create database tables: {e}")
            raise
    
    def store_block(self, block_data: BlockData) -> bool:
        """
        Store block data in the database.
        
        Args:
            block_data: BlockData object to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            # Insert block
            cursor.execute("""
                INSERT OR REPLACE INTO blocks 
                (number, hash, timestamp, gas_used, gas_limit, transaction_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                block_data.number,
                block_data.hash,
                block_data.timestamp,
                block_data.gas_used,
                block_data.gas_limit,
                block_data.transaction_count
            ))
            
            # Store all transactions
            for tx in block_data.transactions:
                self.store_transaction(tx)
            
            self.connection.commit()
            self.logger.info(f"Stored block {block_data.number} with {len(block_data.transactions)} transactions")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store block {block_data.number}: {e}")
            self.connection.rollback()
            return False
    
    def store_transaction(self, tx_data: TransactionData) -> bool:
        """
        Store transaction data in the database.
        
        Args:
            tx_data: TransactionData object to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            # Insert transaction
            cursor.execute("""
                INSERT OR REPLACE INTO transactions 
                (hash, block_number, transaction_index, from_address, to_address, 
                 value, gas, gas_price, gas_used, status, input_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tx_data.hash,
                tx_data.block_number,
                tx_data.transaction_index,
                tx_data.from_address,
                tx_data.to_address,
                str(tx_data.value),  # Convert to string for large values
                tx_data.gas,
                str(tx_data.gas_price),  # Convert to string for large values
                tx_data.gas_used,
                tx_data.status,
                tx_data.input_data
            ))
            
            # Store transaction logs
            if tx_data.logs:
                for log_index, log in enumerate(tx_data.logs):
                    # Convert HexBytes to hex strings for JSON serialization
                    topics = log.get('topics', [])
                    topics_str = json.dumps([topic.hex() if hasattr(topic, 'hex') else str(topic) for topic in topics])
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO transaction_logs 
                        (transaction_hash, log_index, address, topics, data)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        tx_data.hash,
                        log_index,
                        log.get('address', ''),
                        topics_str,
                        log.get('data', '')
                    ))
            
            # Analyze and store contract interactions
            self._analyze_contract_interactions(tx_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store transaction {tx_data.hash}: {e}")
            return False
    
    def _analyze_contract_interactions(self, tx_data: TransactionData):
        """Analyze and store contract interactions for a transaction."""
        cursor = self.connection.cursor()
        
        # Determine interaction type
        if tx_data.to_address is None:
            # Contract creation
            interaction_type = 'create'
            contract_address = 'pending'  # Would need receipt to get actual address
            function_selector = None
        elif tx_data.input_data and tx_data.input_data != '0x':
            # Contract call
            interaction_type = 'call'
            contract_address = tx_data.to_address
            function_selector = tx_data.input_data[:10] if len(tx_data.input_data) >= 10 else None
        else:
            # Simple transfer
            interaction_type = 'transfer'
            contract_address = tx_data.to_address
            function_selector = None
        
        # Store contract interaction
        cursor.execute("""
            INSERT OR REPLACE INTO contract_interactions 
            (transaction_hash, contract_address, function_selector, interaction_type)
            VALUES (?, ?, ?, ?)
        """, (
            tx_data.hash,
            contract_address,
            function_selector,
            interaction_type
        ))
    
    def get_block(self, block_number: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve block data from database.
        
        Args:
            block_number: Block number to retrieve
            
        Returns:
            Block data dictionary or None if not found
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM blocks WHERE number = ?", (block_number,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            self.logger.error(f"Failed to get block {block_number}: {e}")
            return None
    
    def get_transactions_by_block(self, block_number: int) -> List[Dict[str, Any]]:
        """
        Retrieve all transactions for a block.
        
        Args:
            block_number: Block number
            
        Returns:
            List of transaction dictionaries
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM transactions 
                WHERE block_number = ? 
                ORDER BY transaction_index
            """, (block_number,))
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Failed to get transactions for block {block_number}: {e}")
            return []
    
    def get_contract_interactions(self, contract_address: str) -> List[Dict[str, Any]]:
        """
        Get all interactions with a specific contract.
        
        Args:
            contract_address: Contract address
            
        Returns:
            List of interaction dictionaries
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT ci.*, t.block_number, t.gas_used 
                FROM contract_interactions ci
                JOIN transactions t ON ci.transaction_hash = t.hash
                WHERE ci.contract_address = ?
                ORDER BY t.block_number, t.transaction_index
            """, (contract_address,))
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Failed to get contract interactions for {contract_address}: {e}")
            return []
    
    def store_dependency(self, dependent_tx: str, dependency_tx: str, 
                        dependency_type: str, reason: str, gas_impact: int) -> bool:
        """
        Store a transaction dependency relationship.
        
        Args:
            dependent_tx: Hash of dependent transaction
            dependency_tx: Hash of dependency transaction
            dependency_type: Type of dependency
            reason: Reason for dependency
            gas_impact: Gas used by dependent transaction
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO transaction_dependencies 
                (dependent_tx_hash, dependency_tx_hash, dependency_type, dependency_reason, gas_impact)
                VALUES (?, ?, ?, ?, ?)
            """, (dependent_tx, dependency_tx, dependency_type, reason, gas_impact))
            self.connection.commit()
            return True
        except Exception as e:
            self.logger.error(f"Failed to store dependency: {e}")
            return False
    
    def get_dependencies_for_block(self, block_number: int) -> List[Dict[str, Any]]:
        """
        Get all dependencies for transactions in a block.
        
        Args:
            block_number: Block number
            
        Returns:
            List of dependency dictionaries
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT td.*, t1.transaction_index as dependent_index, t2.transaction_index as dependency_index
                FROM transaction_dependencies td
                JOIN transactions t1 ON td.dependent_tx_hash = t1.hash
                JOIN transactions t2 ON td.dependency_tx_hash = t2.hash
                WHERE t1.block_number = ? AND t2.block_number = ?
                ORDER BY t1.transaction_index
            """, (block_number, block_number))
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Failed to get dependencies for block {block_number}: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            cursor = self.connection.cursor()
            
            stats = {}
            
            # Block count
            cursor.execute("SELECT COUNT(*) FROM blocks")
            stats['block_count'] = cursor.fetchone()[0]
            
            # Transaction count
            cursor.execute("SELECT COUNT(*) FROM transactions")
            stats['transaction_count'] = cursor.fetchone()[0]
            
            # Contract interaction count
            cursor.execute("SELECT COUNT(*) FROM contract_interactions")
            stats['contract_interaction_count'] = cursor.fetchone()[0]
            
            # Dependency count
            cursor.execute("SELECT COUNT(*) FROM transaction_dependencies")
            stats['dependency_count'] = cursor.fetchone()[0]
            
            # Block range
            cursor.execute("SELECT MIN(number), MAX(number) FROM blocks")
            min_block, max_block = cursor.fetchone()
            stats['block_range'] = {'min': min_block, 'max': max_block}
            
            # Database file size
            stats['db_file_size'] = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")
    
    def has_block_analysis(self, block_number: int) -> bool:
        """Check if a block has been analyzed (has block data stored)."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1 FROM blocks WHERE number = ? LIMIT 1", (block_number,))
            result = cursor.fetchone()
            return result is not None
        except Exception as e:
            self.logger.error(f"Error checking if block {block_number} exists: {e}")
            return False 