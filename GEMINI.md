# GEMINI.md

This file provides guidance to the Gemini agent when working with the PatternScout repository.

## Project Overview

PatternScout is a tool for automatically detecting "flag" and "pennant" technical analysis patterns from OHLCV financial data.

**Core Features:**
- Read OHLCV data from CSV or MongoDB.
- Automatically detect flag and pennant patterns.
- Generate visualization charts and analysis reports for detected patterns.
- Employ a multi-timeframe architecture to apply adaptive detection parameters.
- Manage datasets of detected patterns and perform breakthrough analysis.

## Development Environment & Commands

The project uses `uv` for environment and package management.

### Setup
```bash
# Install dependencies from uv.lock
uv sync
```

### Running the Application
```bash
# Run the main application
uv run python main.py

# Run a basic scan for a specific symbol
uv run python main.py --symbols RBL8

# Specify a configuration file
uv run python main.py --config config_multi_timeframe.yaml

# Specify pattern types and minimum confidence level
uv run python main.py --pattern-types flag pennant --min-confidence 0.4

# Export the dataset and run breakthrough analysis
uv run python main.py --export-dataset json --analyze-breakthrough
```

### Testing
```bash
# Run the full test suite
uv run python tests/run_tests.py

# Run a specific test module with pytest
uv run python -m pytest tests/test_pattern_detectors.py -v
```

### Code Quality & Linting
```bash
# Check for linting errors with Ruff
uv run ruff check .

# Format code with Ruff
uv run ruff format .

# Run static type checking with MyPy
uv run mypy .
```

## Core Architecture

### Multi-Timeframe System
The system uses a multi-timeframe adaptive architecture to apply the optimal detection parameters based on the data's intrinsic timeframe.

- **TimeframeManager**: Automatically detects the data's timeframe by analyzing the median interval between timestamps. **It does not rely on filenames.**
- **Timeframe Categories**:
  - `ultra_short`: 1m, 3m, 5m
  - `short`: 15m, 30m, 1h
  - `medium_long`: 4h, 1d, 1w
- **Strategy Pattern**: A specific detection strategy (`UltraShortStrategy`, `ShortStrategy`, `MediumLongStrategy`) is selected based on the detected timeframe category. Each strategy has optimized parameters for its respective timeframe.

### Data Flow
1.  **DataConnector**: Fetches OHLCV data from CSV or MongoDB.
2.  **TimeframeManager**: Detects the timeframe.
3.  **PatternDetector**: Detects patterns using the appropriate strategy.
4.  **QualityScorer**: Scores the quality of detected patterns based on multiple dimensions.
5.  **BreakthroughAnalyzer**: Analyzes the effectiveness of patterns post-detection.
6.  **ChartGenerator**: Creates TradingView-style charts for visualization.
7.  **DatasetManager**: Manages pattern data persistence using SQLite and JSON.

### Key Components
- **Detectors**: `BasePatternDetector` provides a common interface for `FlagDetector` and `PennantDetector`.
- **Data Models**: Pydantic and dataclasses define strict types for `PatternRecord`, `Flagpole`, `TrendLine`, etc.
- **Configuration**: Managed via `config.yaml` (standard) and `config_multi_timeframe.yaml`. Parameters are structured by timeframe category.

## Important Notes

### Data Format Requirements
Input CSV files must contain the following columns:
- `timestamp` or `date`
- `symbol`
- `open`, `high`, `low`, `close`
- `volume`

### TA-Lib Dependency
The project requires a specific pre-compiled wheel for TA-Lib on Windows, as defined in `pyproject.toml`.

### Timeframe Detection
The system's ability to auto-detect the timeframe from the data itself is a core feature. When debugging, check the `Auto-detected timeframe` log message in `logs/pattern_scout.log` to confirm the correct strategy is being used.

### Output File Structure
All output is saved to the `output/` directory, which is organized as follows:
```
output/
├── charts/
│   ├── flag/
│   └── pennant/
├── data/
│   ├── patterns/      # Individual pattern data (JSON)
│   ├── patterns.db    # SQLite database
│   └── exports/       # Exported datasets
└── reports/           # Execution reports
```

### Environment
- **Python Version**: Requires Python 3.13 or newer.
- **Environment Variables**: If using MongoDB, copy `.env.example` to `.env` and fill in the connection details.

## Project Structure
The `src/` directory follows a layered architecture:
```
src/
├── data/           # Data access layer (connectors, models)
├── patterns/       # Pattern detection logic (base components, detectors, strategies)
├── analysis/       # Post-detection analysis (breakthroughs)
├── visualization/  # Chart generation
├── storage/        # Data persistence (dataset manager)
└── utils/          # Utility classes (config manager)
```