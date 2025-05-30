#!/usr/bin/env python3
"""
Monitor Continuous Collection
Simple script to check the status and progress of continuous collection.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from storage.database import BlockchainDatabase


def get_collection_status():
    """Get current collection status."""
    database = BlockchainDatabase()
    
    try:
        stats = database.get_database_stats()
        
        if stats.get('block_count', 0) == 0:
            print("❌ No data found - continuous collection may not be running")
            return
        
        # Get latest block info
        cursor = database.connection.cursor()
        cursor.execute("""
            SELECT number, timestamp, created_at 
            FROM blocks 
            ORDER BY number DESC 
            LIMIT 1
        """)
        latest_block = cursor.fetchone()
        
        # Get collection rate (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        cursor.execute("""
            SELECT COUNT(*) 
            FROM blocks 
            WHERE created_at > ?
        """, (one_hour_ago.isoformat(),))
        blocks_last_hour = cursor.fetchone()[0]
        
        # Get recent activity
        cursor.execute("""
            SELECT COUNT(*) 
            FROM blocks 
            WHERE created_at > ?
        """, ((datetime.now() - timedelta(minutes=10)).isoformat(),))
        blocks_last_10min = cursor.fetchone()[0]
        
        print("📊 Continuous Collection Status")
        print("=" * 40)
        print(f"📈 Total Data Collected:")
        print(f"   Blocks: {stats.get('block_count', 0):,}")
        print(f"   Transactions: {stats.get('transaction_count', 0):,}")
        print(f"   Dependencies: {stats.get('dependency_count', 0):,}")
        print(f"   Database size: {stats.get('db_file_size', 0) / 1024 / 1024:.1f} MB")
        
        block_range = stats.get('block_range', {})
        if block_range:
            print(f"\n📦 Block Range:")
            print(f"   First: {block_range.get('min', 'N/A')}")
            print(f"   Latest: {block_range.get('max', 'N/A')}")
            print(f"   Range: {block_range.get('max', 0) - block_range.get('min', 0)} blocks")
        
        if latest_block:
            latest_num, latest_ts, created_at = latest_block
            block_time = datetime.fromtimestamp(latest_ts)
            collection_time = datetime.fromisoformat(created_at)
            
            print(f"\n🕐 Latest Activity:")
            print(f"   Latest block: {latest_num}")
            print(f"   Block timestamp: {block_time}")
            print(f"   Collected at: {collection_time}")
            print(f"   Age: {datetime.now() - collection_time}")
        
        print(f"\n⚡ Collection Rate:")
        print(f"   Last hour: {blocks_last_hour} blocks")
        print(f"   Last 10 min: {blocks_last_10min} blocks")
        
        # Determine status
        if blocks_last_10min > 0:
            status = "🟢 ACTIVE"
        elif blocks_last_hour > 0:
            status = "🟡 SLOW"
        else:
            status = "🔴 STALLED"
        
        print(f"\n📊 Status: {status}")
        
        if blocks_last_10min == 0:
            print("\n💡 Tips:")
            print("   - Check if continuous collection is still running")
            print("   - Verify node connection")
            print("   - Check logs for errors")
        
        # Calculate parallelization stats
        if stats.get('transaction_count', 0) > 0 and stats.get('dependency_count', 0) >= 0:
            parallelization = (1 - stats.get('dependency_count', 0) / stats.get('transaction_count', 0)) * 100
            print(f"\n📈 Overall Parallelization: {parallelization:.1f}%")
    
    finally:
        database.close()


def show_recent_activity():
    """Show recent collection activity."""
    database = BlockchainDatabase()
    
    try:
        cursor = database.connection.cursor()
        cursor.execute("""
            SELECT number, transaction_count, created_at
            FROM blocks 
            ORDER BY number DESC 
            LIMIT 10
        """)
        recent_blocks = cursor.fetchall()
        
        if recent_blocks:
            print("\n📋 Recent Activity (Last 10 blocks):")
            print("   Block     | Txs  | Collected At")
            print("   " + "-" * 35)
            for block_num, tx_count, created_at in recent_blocks:
                collection_time = datetime.fromisoformat(created_at)
                print(f"   {block_num:8} | {tx_count:4} | {collection_time.strftime('%H:%M:%S')}")
    
    finally:
        database.close()


def main():
    """Main monitoring function."""
    try:
        get_collection_status()
        show_recent_activity()
        
        print(f"\n🔄 To run continuous analysis:")
        print(f"   python analyze_continuous_data.py")
        print(f"\n👀 To analyze specific blocks:")
        print(f"   python demo_clean_visualization.py --block <block_number>")
        
    except Exception as e:
        print(f"❌ Error monitoring collection: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 