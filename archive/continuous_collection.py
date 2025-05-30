#!/usr/bin/env python3
"""
Continuous Collection Script
Continuously monitors and analyzes recent Ethereum blocks for transaction dependencies.
"""

import logging
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.ethereum_client import EthereumClient
from storage.database import BlockchainDatabase
from analysis.continuous_collector import ContinuousCollector


def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/continuous_collection.log')
        ]
    )


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """Main entry point for continuous collection."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize components
        logger.info("Initializing Ethereum client...")
        client = EthereumClient(config['ethereum']['rpc_url'])
        
        # Test connection
        chain_id = client.get_chain_id()
        latest_block = client.get_latest_block_number()
        logger.info(f"Connected to Ethereum (Chain ID: {chain_id}, Latest Block: {latest_block})")
        
        logger.info("Initializing database...")
        database = BlockchainDatabase()
        
        # Get collection settings from config
        collection_config = config.get('collection', {})
        lookback_blocks = collection_config.get('lookback_blocks', 100)
        check_interval = collection_config.get('check_interval', 60)
        max_workers = collection_config.get('max_workers', 4)
        
        logger.info("Initializing continuous collector...")
        collector = ContinuousCollector(
            client=client,
            database=database,
            lookback_blocks=lookback_blocks,
            check_interval=check_interval,
            max_workers=max_workers
        )
        
        # Start collection
        logger.info("Starting continuous collection...")
        logger.info("Press Ctrl+C to stop gracefully")
        
        collector.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("Continuous collection stopped")


if __name__ == "__main__":
    main() 