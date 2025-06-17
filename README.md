# Enhanced Stock Recommender System

This project implements a content-based stock recommendation system with portfolio analysis and diversification recommendations, designed to provide personalized stock recommendations with simple, jargon-free explanations.

## Key Features

- **Content-based stock recommendations** using cosine similarity
- **Portfolio risk analysis** with sector concentration alerts
- **Diversification recommendations** from sectors not in your portfolio
- **Simple, jargon-free explanations** for all recommendations
- **Support for consolidated user portfolio formats**
- **Portfolio analysis** with sector distribution and valuation
- **Command-line interfaces** for different use cases

## Files

- `improved_recommender.py` - Enhanced recommender system with diversification and explanation features
- `train_improved_recommender.py` - Script for training the recommender system
- `stock_advisor.py` - Unified command-line interface for all features
- `improved_cli.py` - Full-featured command-line interface
- `diversify_cli.py` - Specialized CLI focused on portfolio diversification and simple explanations
- `test_new_features.py` - Test script for the new features

## Data Format

The system requires two main data files:

1. `stocks_data.csv` - Stock feature data with columns:
   - ticker
   - company_name
   - sector
   - industry 
   - price
   - various financial metrics (pe_ratio, market_cap, etc.)
   - esg_score

2. `users_unique_portfolio.csv` - Consolidated user portfolios with columns:
   - user_id
   - ticker (list of stock tickers as string)
   - weight (list of corresponding weights as string)

## Usage

### Unified Command Line Interface

The easiest way to use the system is with the unified command-line interface:

```
# Analyze a user portfolio with risk assessment
python stock_advisor.py portfolio user_500

# Get information about a specific stock
python stock_advisor.py stock AAPL

# Find similar stocks
python stock_advisor.py similar AAPL --count 3

# Explore stocks in a sector
python stock_advisor.py sector Technology --count 5

# Train or retrain the model
python stock_advisor.py train
```

### Training the Recommender

To train the recommender system:

```
python train_improved_recommender.py
```

### Portfolio Analysis and Diversification

For analyzing portfolio concentration and diversification:

```
python diversify_cli.py analyze user_500
```

This will:
- Display sector allocation with visual indicators
- Check for concentration in specific sectors
- Provide diversification recommendations when needed
- Alert you to potential portfolio risks

### Simple Stock Explanations

For simple explanations of a specific stock:

```
python diversify_cli.py explain --ticker AAPL
```

This provides a jargon-free explanation of:
- What the company does
- Its size and sector
- Its price relative to other stocks
- Its volatility and risk level
- Its environmental and social practices

### User Recommendations with Explanations

For recommendations with simple explanations:

```
python diversify_cli.py explain --user-id user_100 --count 3
```

This provides:
- Personalized stock recommendations based on portfolio
- Simple explanations for why each stock is recommended
- Diversification recommendations if the portfolio is too concentrated

### Full CLI Features

The full-featured CLI supports more advanced operations:

```
python improved_cli.py similar AAPL --count 5   # Find similar stocks
python improved_cli.py analyze user_1           # Full portfolio analysis
python improved_cli.py recommend user_500       # Get recommendations
python improved_cli.py sector Technology        # Analyze a sector
```

## New Features

### 1. Portfolio Risk Analysis

The system now analyzes your portfolio for:
- Over-concentration in specific sectors (>50%)
- Lack of sector diversity (<3 sectors)
- High-beta stock concentration
- Provides actionable alerts with clear explanations

### 2. Diversification Recommendations

When your portfolio is too concentrated, the system:
- Identifies missing sectors in your portfolio
- Recommends high-quality stocks from those sectors
- Explains how each recommendation improves diversification
- Avoids recommending sectors already in your portfolio

### 3. Simple Explanations

All recommendations now include simple, jargon-free explanations:
- Uses plain language a child could understand
- Explains why a stock is recommended in context of your portfolio
- Provides relevant information on price, size, risk and ESG factors
- Avoids complex financial terms

## Implementation Details

The recommender system uses content-based filtering with additional features:

1. **Portfolio Sector Analysis**: Calculates sector concentrations and identifies imbalances
2. **Diversification Algorithm**: Finds stocks from sectors not in your portfolio
3. **Simple Explanation Generator**: Creates easy-to-understand explanations for all recommendations
4. **Risk Assessment**: Identifies portfolio risks and provides actionable alerts

## Future Improvements

- Add hybrid recommendation approach with collaborative filtering
- Implement time-based features for dynamic recommendations
- Add risk-adjusted portfolio optimization
- Implement web interface with visualization dashboard 