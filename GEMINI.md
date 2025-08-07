# GEMINI.md

This file provides guidance to the Gemini agent when working with the PatternScout repository.

## Project Overview

PatternScout is a high-precision flag pattern recognition tool that uses a dynamic baseline system. Key features include:
- **Dynamic Baseline System**: 500K-line rolling statistics, three-layer robust statistical protection, and intelligent market state detection.
- **Failure Signal Filtering**: Fake breakout detection, volume divergence analysis, and pattern deformation monitoring.
- **Six-Category Outcome Tracking**: Strong continuation, standard continuation, breakout stagnation, fake breakout reversal, internal collapse, and reverse run.
- **Multi-Data Source Support**: Reads OHLCV data from CSV or MongoDB data sources.
- **Intelligent Pattern Recognition**: Flag (flag) and pennant (pennant) detection with dynamic threshold adjustment.
- **Enhanced Visualization**: Dynamic baseline charts, failure signal markers, and outcome analysis charts.
- **Multi-Timeframe**: Adaptive detection parameters supporting all cycles from minutes to months.
- **Data Archiving System**: Pattern dataset management and statistical analysis.

## Development Environment & Commands

The project uses `uv` for environment and package management.

### Setup & Basic Commands
```bash
# Install dependencies
uv sync

# Run the main application
uv run python main.py

# Run the full test suite
uv run python tests/run_tests.py
```

### Running Specific Scans & Analyses
```bash
# Dynamic baseline mode scan (recommended)
uv run python main.py --symbols RBL8

# Full dynamic scan with outcome tracking enabled
uv run python main.py --symbols AAPL MSFT --start-date 2023-01-01

# Dynamic scan with outcome tracking disabled
uv run python main.py --symbols RBL8 --disable-outcome-tracking

# Specify a configuration file
uv run python main.py --config config_multi_timeframe.yaml

# Specify pattern types and minimum confidence level
uv run python main.py --pattern-types flag pennant --min-confidence 0.4

# Limit the scan to a specific date range
uv run python main.py --start-date 2023-01-01 --end-date 2023-12-31

# Use MongoDB as the data source
uv run python main.py --data-source mongodb

# Generate baseline summary and outcome charts (skips individual charts)
uv run python main.py --symbols RBL8 --no-charts
```

### Testing
```bash
# Run core test modules
uv run python -m pytest tests/test_technical_indicators.py -v
uv run python -m pytest tests/test_pattern_detectors.py -v

# Run dynamic baseline system tests
uv run python -m pytest tests/test_dynamic_baseline_system.py -v
uv run python -m pytest tests/test_dynamic_flagpole_detector.py -v
uv run python -m pytest tests/test_dynamic_flag_detector.py -v
uv run python -m pytest tests/test_pattern_outcome_tracker.py -v
uv run python -m pytest tests/test_dynamic_pattern_scanner.py -v

# Run specialized algorithm tests
uv run python -m pytest tests/test_atr_adaptive.py -v
uv run python -m pytest tests/test_ransac.py -v

# Run performance and validation tests
uv run python tests/quick_real_test.py
uv run python tests/simple_performance_analysis.py
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

### Dynamic Baseline System Architecture
PatternScout uses an advanced baseline system that completely refactors the pattern recognition process:

#### Six-Stage Recognition and Data Archiving Process
- **Stage 0 - Dynamic Baseline System**: 500K-line rolling statistics, three-layer robust statistical protection, and intelligent market state detection.
- **Stage 1 - Dynamic Flagpole Detection**: Flagpole recognition based on dynamic thresholds, slope scoring, and volume burst verification.
- **Stage 2 - Dynamic Flag Detection**: Percentile channel construction, failure signal pre-filtering, and geometric pattern analysis.
- **Stage 3 - Pattern Outcome Analysis**: Six-category outcome monitoring system with pure outcome classification logic.
- **Stage 4 - Pattern Data Output**: Structured data recording, pattern DNA feature storage, and environment snapshot archiving.
- **Stage 5 - Pattern Visualization and Chart Output**: Standardized PNG format K-line charts and "visual fingerprint" generation for patterns.

#### Core Technical Features
- **Three-Layer Robust Statistical Protection**: MAD filtering + Winsorize + dynamic threshold adjustment to prevent contamination from outliers.
- **Intelligent Market State Detection**: Dual-state baseline management, anti-oscillation mechanism, and adaptive volatility analysis.
- **Failure Signal Recognition**: Fake breakout detection, volume divergence analysis, and pattern deformation monitoring.
- **Six-Category Outcome System**: Strong continuation, standard continuation, breakout stagnation, fake breakout reversal, internal collapse, and reverse run.

### Unified Pattern Detection Architecture
- **Unified Pattern Type**: `PatternType.FLAG_PATTERN` includes two subtypes:
  - `FlagSubType.FLAG`: Rectangular flag (parallel channel)
  - `FlagSubType.PENNANT`: Pennant (converging triangle)
- **Single Detector**: `FlagDetector` handles all flag variations uniformly.
- **Backward Compatibility**: `PatternScanner` provides legacy `scan_flags()` and `scan_pennants()` methods.

### Multi-Timeframe Architecture
- **TimeframeManager**: Auto-detects data timeframe by analyzing timestamp intervals, not filenames.
- **Timeframe Categories**: 1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M.
- **Strategy Pattern**: Uses specially optimized detection strategies for different timeframes.
- **Adaptive Parameters**: Automatically selects parameter sets based on the detected timeframe.

### Data Flow
1.  **DataConnector**: Fetches OHLCV data from CSV or MongoDB.
2.  **TimeframeManager**: Detects the timeframe.
3.  **FlagDetector**: Identifies all flag variations using a unified detector.
4.  **QualityScorer**: Performs multi-dimensional quality scoring.
5.  **BreakthroughAnalyzer**: Analyzes pattern effectiveness.
6.  **ChartGenerator**: Creates TradingView-style charts.
7.  **DatasetManager**: Manages data persistence.


### Key Components & Design Patterns
- **Detectors**: `DynamicPatternScanner` coordinates the execution of the entire six-stage process.
- **Data Models**: Pydantic and dataclasses define strict types for `PatternRecord`, `DynamicBaseline`, `InvalidationSignal`, etc.
- **Configuration**: Managed via `config.yaml` and `config_multi_timeframe.yaml`.
- **Factory Pattern**: `DataConnectorFactory` creates the appropriate data connector (CSV/MongoDB).
- **Strategy Pattern**: Timeframe-specific strategies (`MinuteOneStrategy`, etc.) encapsulate detection logic.

## Algorithm Details & Optimizations

- **Unified Pattern Detection**: A single `FlagDetector` handles both rectangular flags and pennants, using overlap detection and confidence optimization.
- **Pennant Convergence**: Calculates the apex and convergence ratio of trend lines.
- **Flagpole Detection**: Based on price change rate and volume spikes.
- **Quality Scoring**: Multi-dimensional weighted scoring (geometric features, volume patterns, technical confirmation).
- **Flag Parallelism**: Uses the R² value from linear regression to verify the flag's channel quality.

### Algorithm Optimizations (Implemented Jan 2025)
- **ATR-Adaptive Parameter System**: Automatically adjusts detection parameters based on market volatility (ATR).
- **RANSAC Trend Line Fitting**: Provides robust outlier handling for trend lines.
- **Intelligent Swing Point Detection**: Uses an ATR-based prominence filter instead of simple local extrema.
- **Enhanced Volume Analysis**: Incorporates multi-dimensional volume health scoring.
- **Improved Flag/Pennant Logic**: Includes channel divergence checks, breakout readiness assessment, and support for different triangle classifications.

## Important Notes

### Data Format Requirements
Input CSV files must contain: `timestamp` or `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`.

### TA-Lib Dependency
Requires a specific pre-compiled wheel for TA-Lib on Windows, as defined in `pyproject.toml`.

### Timeframe Detection
The system auto-detects the timeframe from the data itself. Check the `Auto-detected timeframe` log message in `logs/pattern_scout.log`.

### Output File Structure
```
output/
├── data/
│   ├── patterns/          # JSON pattern records
│   ├── patterns.db        # SQLite database
│   └── exports/           # Exported files
├── charts/
│   ├── flag/              # Flag charts
│   ├── pennant/           # Pennant charts
│   └── summary/           # Summary charts
└── reports/               # Execution reports
```

### Environment
- **Python Version**: Requires Python 3.13 or newer.
- **Environment Variables**: If using MongoDB, copy `.env.example` to `.env` and fill in the connection details.

## Project Structure (PatternScout 3.0 Refactoring)
```
src/
├── data/           # Data layer
│   ├── connectors/ # CSV/MongoDB connectors
│   └── models/     # Pydantic data models (enhanced: invalidation signals, outcome analysis, market snapshots)
├── patterns/       # Pattern detection layer (Stages 0-2)
│   ├── base/       # Base components
│   │   ├── robust_statistics.py      # Three-layer robust statistical protection
│   │   ├── market_regime_detector.py # Intelligent market state detection
│   │   └── timeframe_manager.py      # Timeframe management
│   ├── detectors/  # Detector implementations
│   │   ├── dynamic_flagpole_detector.py # Stage 1: Dynamic flagpole detector
│   │   ├── dynamic_flag_detector.py     # Stage 2: Dynamic flag detector
│   │   ├── dynamic_pattern_scanner.py  # Six-stage unified scanner
│   │   └── pattern_scanner.py          # Legacy scanner (for compatibility)
│   ├── indicators/ # Technical indicators
│   └── strategies/ # Timeframe strategies
├── analysis/       # Outcome analysis layer (Stage 3)
│   ├── pattern_outcome_analyzer.py    # Pure outcome analyzer
│   └── pattern_outcome_tracker.py     # Legacy outcome tracker (for compatibility)
├── storage/        # Data output layer (Stage 4)
│   ├── pattern_data_exporter.py       # Pattern data exporter
│   └── dataset_manager.py             # Legacy dataset manager (for compatibility)
├── visualization/  # Visualization layer (Stage 5)
│   ├── pattern_chart_generator.py     # Specialized pattern chart generator
│   ├── chart_generator.py             # Legacy chart generator (for compatibility)
│   └── dynamic_chart_methods.py       # Dynamic baseline chart methods
└── utils/          # Utility classes (config management)
```

## Refactoring Description (PatternScout 3.0)

This version introduces a major refactoring with the six-stage dynamic baseline system.

- **Core System Refactoring**: Implemented the six-stage process from baseline updates to chart generation.
- **Data Model Expansion**: Added new enums and data structures for outcomes, market regimes, and invalidation signals.
- **Modular Architecture**: Each stage is a configurable and independent module.
- **API Compatibility**: Full backward compatibility with previous APIs.
- **Comprehensive Testing**: The test suite covers all components of the new six-stage system.
