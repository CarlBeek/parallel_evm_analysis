"""
Collect commands for the parallel-stats CLI.
Handles continuous data collection operations.
"""

import logging
import sys
import yaml
from pathlib import Path

# Import the existing functionality from organized modules
from core.ethereum_client import EthereumClient
from storage.database import BlockchainDatabase
from analysis.continuous_collector import ContinuousCollector


def handle_collect(args):
    """Handle all collect subcommands."""
    if args.collect_cmd == 'start':
        return start_collection(args.workers, args.interval)
    elif args.collect_cmd == 'monitor':
        return monitor_collection()
    elif args.collect_cmd == 'stop':
        return stop_collection()
    else:
        print("❌ No collect operation specified. Use --help for options.")
        return 1


def start_collection(workers=4, interval=60):
    """Start continuous data collection."""
    print("🚀 Starting continuous data collection...")
    
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
        print(f"🔗 Connected to Ethereum (Chain ID: {chain_id}, Latest Block: {latest_block})")
        
        logger.info("Initializing database...")
        database = BlockchainDatabase()
        
        # Get collection settings
        collection_config = config.get('collection', {})
        lookback_blocks = collection_config.get('lookback_blocks', 100)
        
        print(f"⚙️  Configuration:")
        print(f"   Workers: {workers}")
        print(f"   Check interval: {interval}s")
        print(f"   Lookback blocks: {lookback_blocks}")
        
        logger.info("Initializing continuous collector...")
        collector = ContinuousCollector(
            client=client,
            database=database,
            lookback_blocks=lookback_blocks,
            check_interval=interval,
            max_workers=workers
        )
        
        # Start collection
        print("🔄 Starting continuous collection...")
        print("   Press Ctrl+C to stop gracefully")
        
        collector.start()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Collection stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error starting collection: {e}")
        logger.error(f"Fatal error: {e}")
        return 1
    finally:
        print("🛑 Continuous collection stopped")


def monitor_collection():
    """Monitor collection status and progress."""
    print("📊 Monitoring collection status...")
    
    try:
        database = BlockchainDatabase()
        
        # Get collection status
        print("🔍 Collection Status:")
        get_collection_status(database)
        
        print("\n📈 Recent Activity:")
        show_recent_activity(database)
        
        print(f"\n💡 Commands:")
        print(f"   🔄 To start collection: parallel-stats collect start")
        print(f"   📊 To analyze data: parallel-stats analyze continuous")
        print(f"   🛑 To stop collection: parallel-stats collect stop")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error monitoring collection: {e}")
        return 1
    finally:
        try:
            database.close()
        except:
            pass


def stop_collection():
    """Stop continuous collection gracefully."""
    print("🛑 Stopping continuous collection...")
    
    # This would implement graceful shutdown
    # For now, just provide instructions
    print("ℹ️  To stop collection, use Ctrl+C in the collection terminal")
    print("   or kill the collection process")
    
    return 0


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
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Return default configuration
        return {
            'ethereum': {
                'rpc_url': 'http://ts-geth:8545'
            },
            'collection': {
                'lookback_blocks': 100,
                'check_interval': 60,
                'max_workers': 4
            }
        }


def get_collection_status(database):
    """Get and display collection status."""
    try:
        stats = database.get_database_stats()
        
        block_count = stats.get('block_count', 0)
        dependency_count = stats.get('dependency_count', 0)
        block_range = stats.get('block_range', {})
        
        if block_count == 0:
            print("   ❌ No data collected yet")
        else:
            print(f"   ✅ {block_count} blocks collected")
            print(f"   📊 {dependency_count} dependencies found")
            
            if block_range:
                min_block = block_range.get('min_block', 'Unknown')
                max_block = block_range.get('max_block', 'Unknown')
                print(f"   📈 Block range: {min_block} - {max_block}")
                
    except Exception as e:
        print(f"   ❌ Error getting status: {e}")


def show_recent_activity(database):
    """Show recent collection activity."""
    try:
        # This would show recent blocks and analysis results
        # For now, show basic info
        stats = database.get_database_stats()
        
        if stats.get('block_count', 0) > 0:
            print("   📅 Recent collection data available")
            print("   📊 Use 'parallel-stats analyze continuous' for detailed analysis")
        else:
            print("   ℹ️  No recent activity - start collection to begin")
            
    except Exception as e:
        print(f"   ❌ Error showing activity: {e}") 