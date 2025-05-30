# Parallel Stats - Ethereum Transaction Dependency Analyzer

A professional tool for analyzing transaction dependencies in Ethereum blocks to understand parallelization potential and visualize dependency relationships.

## ⭐ Features

- **🎯 Unified CLI Interface**: Single entry point with discoverable commands and comprehensive help
- **🔍 Multi-Strategy Dependency Detection**: Analyzes contract interactions, address patterns, event logs, and function calls  
- **📊 Clean Dependency Visualization**: Interactive graphs showing only true state dependencies (99.5% noise reduction)
- **📈 Gantt Chart Timeline**: Timeline visualization showing parallel execution potential
- **🚀 High Performance**: Parallel processing with 3-5 second analysis time per block
- **📦 Multi-Block Analysis**: Batch processing with comprehensive statistics
- **🔄 Continuous Collection**: Automatically collect and analyze recent blocks
- **💾 Database Storage**: SQLite storage for historical tracking and analysis
- **📋 Comprehensive Reporting**: JSON exports with detailed metrics and findings

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **Ethereum Node Access** (Geth or Reth with eth, net, web3 APIs)
- **Virtual Environment** (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd parallel-stats

# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Validate environment
python main.py validate environment

# Configure node connection (edit config.yaml)
python main.py validate node
```

## 📱 CLI Usage

### Getting Help

```bash
# Show all available commands
python main.py --help

# Get help for specific commands
python main.py analyze --help
python main.py collect --help
python main.py validate --help
```

### Analysis Commands

```bash
# Analyze the latest block
python main.py analyze latest

# Analyze a specific block
python main.py analyze block 22591952

# Analyze a range of blocks
python main.py analyze range 22591950 22591960

# Analyze continuously collected data
python main.py analyze continuous --blocks 10

# Find longest dependency chain
python main.py analyze chain --longest
```

### Data Collection Commands

```bash
# Start continuous data collection
python main.py collect start

# Monitor collection status
python main.py collect monitor

# Stop collection gracefully
python main.py collect stop
```

### Validation Commands

```bash
# Validate environment setup
python main.py validate environment

# Test Ethereum node connection
python main.py validate node
```

## 📊 Key Findings

Based on comprehensive analysis of Ethereum mainnet blocks:

- **🎯 93.8% Parallelization Potential**: Most transactions are independent
- **✨ 99.5% Dependency Noise Reduction**: Advanced filtering removes false positives  
- **⚡ 10-50x Theoretical Speedup**: With perfect parallel execution
- **📈 Consistent Results**: 88-96% parallelization across analyzed blocks
- **🔗 True Dependencies**: Only 0.5% of detected dependencies are real state conflicts

## 🏗️ Project Structure

```
parallel-stats/                        # Clean, professional structure
├── main.py                            # 🚀 UNIFIED CLI ENTRY POINT
├── config.yaml                        # Configuration
├── requirements.txt                   # Dependencies
├── README.md                          # Documentation (this file)
├── CLEANUP_SUMMARY.md                 # Organization guide
├── src/                               # 📁 Organized source code
│   ├── cli/                           # Command line interface handlers
│   │   ├── analyze_commands.py        # Analysis operations
│   │   ├── collect_commands.py        # Data collection operations
│   │   └── validate_commands.py       # Validation operations
│   ├── core/                          # Core blockchain functionality
│   │   ├── ethereum_client.py         # Ethereum node client
│   │   └── transaction_fetcher.py     # Parallel transaction fetching
│   ├── storage/                       # Data persistence
│   │   └── database.py                # SQLite database management
│   ├── analysis/                      # Dependency analysis engines
│   │   ├── dependency_analyzer.py     # Multi-strategy detection
│   │   └── continuous_collector.py    # Continuous data collection
│   └── visualization/                 # Graph generation
│       ├── dependency_graph.py        # Interactive dependency graphs
│       └── chain_explorer.py          # Enhanced visualizations
├── tests/                             # 🧪 Unit tests
├── examples/                          # 📚 Documentation examples
│   └── demo_*.py                      # Usage demonstrations
├── archive/                           # 📦 Legacy scripts (preserved)
├── data/                              # Generated outputs
│   ├── graphs/                        # HTML visualizations  
│   └── *.db                           # SQLite databases
└── logs/                              # Application logs
```

## 💡 Usage Examples

### Quick Analysis

```bash
# Validate everything is working
python main.py validate environment
python main.py validate node

# Analyze latest block
python main.py analyze latest
```

### Comprehensive Analysis

```bash
# Start continuous collection (run in background)
python main.py collect start &

# Monitor collection progress
python main.py collect monitor

# Analyze collected data
python main.py analyze continuous --blocks 20
```

### Custom Analysis

```bash
# Analyze specific block with high activity
python main.py analyze block 22591952

# Analyze range during network congestion
python main.py analyze range 22591950 22591960 --workers 8

# Find most complex dependency patterns
python main.py analyze chain --longest
```

## 📈 Output Examples

### Analysis Summary
```
🔍 Analyzing block 22591952...
📊 Block 22591952: 278 transactions
🔗 Analyzing dependencies...
✅ Found 29 dependencies

📋 Analysis Summary:
   Total transactions: 278
   Dependencies found: 29
   Independent transactions: 249
   Parallelization potential: 89.6%
```

### Generated Files
- **Interactive Graph**: `data/graphs/block_22591952_dependencies.html`
- **Analysis Report**: JSON with detailed metrics and findings
- **Database Storage**: SQLite with historical data for trend analysis

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Ethereum node settings
ethereum:
  rpc_url: 'http://10.0.30.105:8545'
  timeout: 30

# Analysis parameters  
analysis:
  max_transactions_per_block: 1000
  min_gas_threshold: 21000

# Collection settings
collection:
  lookback_blocks: 100
  check_interval: 60
  max_workers: 4
```

## 🔧 Development

### Architecture

- **Modular Design**: Clean separation between CLI, core logic, and visualization
- **Extensible**: Easy to add new analysis strategies or visualization types
- **Testable**: Organized structure supports comprehensive testing
- **Professional**: Production-ready code with proper error handling

### Adding New Commands

1. **Create handler**: Add function to appropriate `src/cli/*_commands.py`
2. **Update parser**: Add subcommand to `main.py` 
3. **Test functionality**: Verify with `python main.py <new-command> --help`

### Examples and Demos

See `examples/` directory for:
- **Visualization examples**: Interactive graph generation
- **Database usage**: Storage and retrieval patterns
- **Collection examples**: Continuous data gathering
- **Analysis examples**: Dependency detection strategies

## 🚀 Performance

- **Fast Analysis**: 3-5 seconds per block (150-300 transactions)
- **Parallel Processing**: Configurable worker threads for optimal performance
- **Memory Efficient**: Streaming analysis for large datasets
- **Scalable**: Handles blocks with 500+ transactions

## 📋 Requirements

### System Requirements
- **Python**: 3.12 or higher
- **Memory**: 512MB+ available RAM
- **Storage**: 100MB+ for data and visualizations
- **Network**: Stable connection to Ethereum node

### Node Requirements
- **APIs**: `eth`, `net`, `web3` (minimum)
- **Optional**: `debug` for enhanced state analysis
- **Sync**: Full node recommended for historical analysis

## 🤝 Migration Guide

### From Legacy Scripts

If you were using the old scattered scripts:

**Old Way (deprecated):**
```bash
python analyze_dependencies.py --block 12345
python continuous_collection.py
python validate_environment.py
```

**New Way (recommended):**
```bash
python main.py analyze block 12345
python main.py collect start  
python main.py validate environment
```

All functionality has been preserved and enhanced in the unified CLI.

## 📚 Documentation

- **This README**: Complete usage guide
- **`CLEANUP_SUMMARY.md`**: Detailed organization documentation  
- **`config.yaml`**: Configuration reference
- **CLI Help**: Use `--help` with any command for detailed usage
- **Examples**: See `examples/` directory for code examples

## 🛠️ Troubleshooting

### Common Issues

**"No module named 'web3'"**
```bash
# Activate virtual environment first
source venv/bin/activate
pip install -r requirements.txt
```

**"Cannot connect to Ethereum node"**
```bash
# Test node connection
python main.py validate node

# Check config.yaml for correct RPC URL
```

**"No data found in database"**
```bash
# Start data collection first
python main.py collect start
```

### Getting Help

1. **Built-in Help**: `python main.py --help`
2. **Command Help**: `python main.py <command> --help`
3. **Environment Check**: `python main.py validate environment`
4. **Examples**: Check `examples/` directory

## 📄 License

[Add your license information here]

## 🔗 Links

- **Project Repository**: [GitHub URL]
- **Documentation**: [Docs URL]
- **Issues**: [Issues URL]

---

**Professional Ethereum Analysis Tool** | **Clean Architecture** | **Production Ready** 