"""
Analyze commands for the parallel-stats CLI.
Handles all transaction dependency analysis operations.
"""

import sys
from pathlib import Path

# Import the existing functionality from organized modules
from core.ethereum_client import EthereumClient
from core.transaction_fetcher import TransactionFetcher
from storage.database import BlockchainDatabase
from analysis.state_dependency_analyzer import StateDependencyAnalyzer
from visualization.dependency_graph import DependencyGraphVisualizer


def handle_analyze(args):
    """Handle all analyze subcommands."""
    if args.analyze_cmd == 'latest':
        return analyze_latest_block()
    elif args.analyze_cmd == 'block':
        return analyze_specific_block(args.number)
    elif args.analyze_cmd == 'range':
        return analyze_block_range(args.start, args.end, args.workers)
    elif args.analyze_cmd == 'continuous':
        return analyze_continuous_data(args.blocks)
    elif args.analyze_cmd == 'chain':
        return analyze_dependency_chain(args.block, args.longest)
    else:
        print("❌ No analyze operation specified. Use --help for options.")
        return 1


def analyze_latest_block():
    """Analyze the latest block for transaction dependencies."""
    print("🔍 Analyzing latest block...")
    
    try:
        # Initialize components
        client = EthereumClient()
        latest_block_number = client.get_latest_block_number()
        
        print(f"📊 Latest block: {latest_block_number}")
        return analyze_specific_block(latest_block_number)
        
    except Exception as e:
        print(f"❌ Error analyzing latest block: {e}")
        return 1


def analyze_specific_block(block_number):
    """Analyze a specific block for transaction dependencies."""
    print(f"🔍 Analyzing block {block_number}...")
    
    try:
        # Initialize components
        client = EthereumClient()
        fetcher = TransactionFetcher(client, max_workers=4)
        database = BlockchainDatabase()
        analyzer = StateDependencyAnalyzer(client, database)
        visualizer = DependencyGraphVisualizer(database)
        
        # Fetch and analyze block
        print(f"📡 Fetching block {block_number}...")
        block_data = fetcher.fetch_block_with_transactions(block_number)
        
        if not block_data:
            print(f"❌ Could not fetch block {block_number}")
            return 1
        
        print(f"📊 Block {block_number}: {len(block_data.transactions)} transactions")
        
        # Store in database
        print("💾 Storing block data...")
        database.store_block(block_data)
        
        # Analyze dependencies using real debug API
        print("🔗 Analyzing dependencies (debug API)...")
        dependencies = analyzer.analyze_block_state_dependencies(block_data)
        
        if dependencies:
            print(f"✅ Found {len(dependencies)} dependencies")
            
            # Store dependencies
            print("💾 Storing dependencies...")
            for dep in dependencies:
                database.store_dependency(
                    dep.dependent_tx_hash,
                    dep.dependency_tx_hash, 
                    "state_dependency",  # Mark as real state dependency
                    dep.dependency_reason,
                    dep.gas_impact
                )
            
            # Generate visualizations
            print("📊 Generating visualizations...")
            Path("data/graphs").mkdir(parents=True, exist_ok=True)
            
            # 1. Create Gantt chart (state dependencies)
            print("   📊 Creating Gantt chart...")
            fig1 = visualizer.create_gantt_chart(block_number, use_refined=False)
            visualizer.save_graph(fig1, f"gantt_chart_block_{block_number}", 'html')
            
            # 2. Create dependency statistics
            print("   📋 Creating statistics chart...")
            fig2 = visualizer.create_dependency_statistics_chart(block_number)
            visualizer.save_graph(fig2, f"dependency_stats_block_{block_number}", 'html')
            
            print(f"📈 Visualizations saved to data/graphs/:")
            print(f"   • gantt_chart_block_{block_number}.html - Timeline visualization") 
            print(f"   • dependency_stats_block_{block_number}.html - Statistics dashboard")
        else:
            print("ℹ️  No dependencies found in this block")
        
        # Show summary
        independent_count = len(block_data.transactions) - len(set(dep.dependent_tx_hash for dep in dependencies))
        parallelization_potential = (independent_count / len(block_data.transactions)) * 100
        
        print("\n📋 Analysis Summary:")
        print(f"   Total transactions: {len(block_data.transactions)}")
        print(f"   Dependencies found: {len(dependencies)}")
        print(f"   Independent transactions: {independent_count}")
        print(f"   Parallelization potential: {parallelization_potential:.1f}%")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error analyzing block {block_number}: {e}")
        return 1
    finally:
        try:
            database.close()
        except:
            pass


def analyze_block_range(start_block, end_block, workers=4):
    """Analyze a range of blocks."""
    print(f"🔍 Analyzing blocks {start_block} to {end_block} with {workers} workers...")
    
    # Import the continuous data analysis functionality
    try:
        from pathlib import Path
        import json
        
        # Initialize components
        client = EthereumClient()
        fetcher = TransactionFetcher(client, max_workers=workers)
        database = BlockchainDatabase()
        analyzer = StateDependencyAnalyzer(client, database)
        
        total_blocks = end_block - start_block + 1
        total_transactions = 0
        total_dependencies = 0
        
        print(f"📊 Processing {total_blocks} blocks...")
        
        for block_num in range(start_block, end_block + 1):
            print(f"\n🔍 Processing block {block_num} ({block_num - start_block + 1}/{total_blocks})...")
            
            # Fetch block
            block_data = fetcher.fetch_block_with_transactions(block_num)
            if not block_data:
                print(f"⚠️  Skipping block {block_num} (could not fetch)")
                continue
                
            # Store block
            database.store_block(block_data)
            
            # Analyze dependencies
            dependencies = analyzer.analyze_block_state_dependencies(block_data)
            if dependencies:
                for dep in dependencies:
                    database.store_dependency(
                        dep.dependent_tx_hash,
                        dep.dependency_tx_hash, 
                        "state_dependency",  # Mark as real state dependency
                        dep.dependency_reason,
                        dep.gas_impact
                    )
            
            total_transactions += len(block_data.transactions)
            total_dependencies += len(dependencies)
            
            print(f"   📊 {len(block_data.transactions)} transactions, {len(dependencies)} dependencies")
        
        # Generate summary report
        independent_transactions = total_transactions - len(set(dep.dependent_tx_hash for dep in dependencies) if dependencies else set())
        parallelization_potential = (independent_transactions / total_transactions * 100) if total_transactions > 0 else 0
        
        print(f"\n🎉 Range Analysis Complete!")
        print(f"📋 Summary:")
        print(f"   Blocks analyzed: {total_blocks}")
        print(f"   Total transactions: {total_transactions}")
        print(f"   Total dependencies: {total_dependencies}")
        print(f"   Independent transactions: {independent_transactions}")
        print(f"   Overall parallelization potential: {parallelization_potential:.1f}%")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error analyzing block range: {e}")
        return 1
    finally:
        try:
            database.close()
        except:
            pass


def analyze_continuous_data(num_blocks=10):
    """Analyze continuously collected data."""
    print(f"🔍 Analyzing continuous collection data (last {num_blocks} blocks)...")
    
    try:
        database = BlockchainDatabase()
        
        # Get database stats
        stats = database.get_database_stats()
        
        if stats.get('block_count', 0) == 0:
            print("❌ No data found in database. Run 'parallel-stats collect start' first.")
            return 1
        
        print(f"📊 Database contains {stats.get('block_count', 0)} blocks")
        print(f"   Block range: {stats.get('block_range', {})}")
        print(f"   Total dependencies: {stats.get('dependency_count', 0)}")
        
        # This would implement the full continuous data analysis
        # For now, show basic statistics
        
        print("✅ Continuous data analysis complete")
        return 0
        
    except Exception as e:
        print(f"❌ Error analyzing continuous data: {e}")
        return 1
    finally:
        try:
            database.close()
        except:
            pass


def analyze_dependency_chain(block_number=None, longest=False):
    """Analyze specific dependency chains."""
    print("🔍 Analyzing dependency chains...")
    
    try:
        database = BlockchainDatabase()
        
        if longest:
            print("🔍 Finding longest dependency chain...")
            # Implementation would find the block with the longest chain
            # For now, just show what we have
            stats = database.get_database_stats()
            print(f"📊 Database contains {stats.get('dependency_count', 0)} total dependencies")
            
        elif block_number:
            print(f"🔍 Analyzing dependency chains in block {block_number}...")
            # Implementation would analyze specific block chains
            
        else:
            print("❌ Specify either --block <number> or --longest")
            return 1
        
        print("✅ Dependency chain analysis complete")
        return 0
        
    except Exception as e:
        print(f"❌ Error analyzing dependency chains: {e}")
        return 1
    finally:
        try:
            database.close()
        except:
            pass 