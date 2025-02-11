# Probabilistic Trading

A probabilistic-based trading system built using nautilus_trader, implementing Hidden Markov Models and Kalman Filters for market analysis and trading decisions.

## Requirements

- Python 3.12 or higher
- uv (modern Python package installer)

## Installation

### 1. Installing uv

uv is a modern Python package installer and resolver that offers significant performance improvements over traditional tools. There are several ways to install uv:

Using curl (recommended for Unix-like systems):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Using PowerShell (for Windows):

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more detailed installation instructions and alternatives, please refer to the [official uv documentation](https://docs.astral.sh/uv/).

### 2. Installing Project(Development)

To install the package for development:

1. Clone the repository:

```bash
git clone https://github.com/Martingale42/probabilistic_trader.git
cd probabilistic_trading
```

2. Install the package in editable mode with development dependencies:

```bash
uv pip install -e ".[dev]"
```

### 3. Installing Project(Stable)

To install the latest version as a user:

```bash
uv pip install git+https://github.com/Martingale42/probabilistic_trader.git
```

## Features

- Hidden Markov Model (HMM) based trading strategies
- Kalman Filter implementation for state estimation
- Integration with nautilus_trader for backtesting and live trading
- Support for multiple timeframes and trading pairs

## Project Structure

```
probabilistic_trading/
├── model/
│   ├── hmm/         # Hidden Markov Model implementations
│   └── kalman/      # Kalman Filter implementations
└── strategy/        # Trading strategies
```

## Usage

Basic example of using the HMM strategy:

```python
from probabilistic_trading.strategy import HMMStrategy

# Example code here
```

## Development

This project uses several development tools:

- black for code formatting
- ruff for linting
- mypy for type checking
- pytest for testing

To run the development tools:

```bash
# Format code
black .

# Run linting
ruff check .

# Run type checking
mypy .

# Run tests
pytest
```