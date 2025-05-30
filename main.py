#!/usr/bin/env python3
"""
Parallel Stats - Ethereum Transaction Dependency Analysis Tool
Single entry point for all blockchain parallelization analysis.

This replaces 23+ scattered scripts with a unified, discoverable interface.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_parser():
    """Create the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog='parallel-stats',
        description='Ethereum Transaction Dependency Analysis Tool',
        epilog='Use "parallel-stats <command> --help" for command-specific help.'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='parallel-stats 1.0.0'
    )
    
    # Create subcommands
    subparsers = parser.add_subparsers(
        dest='command', 
        help='Available commands',
        metavar='<command>'
    )
    
    # ANALYZE command group
    analyze_parser = subparsers.add_parser(
        'analyze', 
        help='Analyze transaction dependencies'
    )
    analyze_subs = analyze_parser.add_subparsers(
        dest='analyze_cmd',
        help='Analysis operations',
        metavar='<operation>'
    )
    
    # analyze latest
    analyze_subs.add_parser(
        'latest',
        help='Analyze the latest block'
    )
    
    # analyze block
    block_parser = analyze_subs.add_parser(
        'block',
        help='Analyze a specific block'
    )
    block_parser.add_argument(
        'number',
        type=int,
        help='Block number to analyze'
    )
    
    # analyze range
    range_parser = analyze_subs.add_parser(
        'range',
        help='Analyze a range of blocks'
    )
    range_parser.add_argument(
        'start',
        type=int,
        help='Starting block number'
    )
    range_parser.add_argument(
        'end',
        type=int,
        help='Ending block number'
    )
    range_parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    # analyze continuous
    continuous_parser = analyze_subs.add_parser(
        'continuous',
        help='Analyze continuously collected data'
    )
    continuous_parser.add_argument(
        '--blocks',
        type=int,
        default=10,
        help='Number of recent blocks to analyze (default: 10)'
    )
    
    # analyze chain
    chain_parser = analyze_subs.add_parser(
        'chain',
        help='Analyze specific dependency chain'
    )
    chain_parser.add_argument(
        '--block',
        type=int,
        help='Block number to analyze'
    )
    chain_parser.add_argument(
        '--longest',
        action='store_true',
        help='Find and analyze the longest dependency chain'
    )
    
    # COLLECT command group
    collect_parser = subparsers.add_parser(
        'collect',
        help='Data collection operations'
    )
    collect_subs = collect_parser.add_subparsers(
        dest='collect_cmd',
        help='Collection operations',
        metavar='<operation>'
    )
    
    # collect start
    start_parser = collect_subs.add_parser(
        'start',
        help='Start continuous data collection'
    )
    start_parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    start_parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Check interval in seconds (default: 60)'
    )
    
    # collect monitor
    collect_subs.add_parser(
        'monitor',
        help='Monitor collection status and progress'
    )
    
    # collect stop
    collect_subs.add_parser(
        'stop',
        help='Stop continuous collection gracefully'
    )
    
    # VALIDATE command group
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validation and testing'
    )
    validate_subs = validate_parser.add_subparsers(
        dest='validate_cmd',
        help='Validation operations',
        metavar='<operation>'
    )
    
    # validate environment
    validate_subs.add_parser(
        'environment',
        help='Validate environment setup'
    )
    
    # validate node
    validate_subs.add_parser(
        'node',
        help='Validate Ethereum node connection'
    )
    
    return parser

def main():
    """Main entry point for the CLI application."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Route to appropriate command handlers
    try:
        if args.command == 'analyze':
            from cli.analyze_commands import handle_analyze
            return handle_analyze(args)
        elif args.command == 'collect':
            from cli.collect_commands import handle_collect
            return handle_collect(args)
        elif args.command == 'validate':
            from cli.validate_commands import handle_validate
            return handle_validate(args)
        else:
            # No command provided, show help
            parser.print_help()
            return 0
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 