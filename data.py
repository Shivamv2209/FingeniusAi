import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_comprehensive_stock_data(num_stocks=500, output_dir="stock_recommender_data"):
    """
    Generate a comprehensive stock dataset for training a recommendation system
    
    Parameters:
    num_stocks (int): Number of stocks to generate
    output_dir (str): Directory to save the generated data
    
    Returns:
    tuple: Paths to the generated CSV files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Define stock tickers - using a mix of real tickers and generated ones
    real_tickers = [
        # Technology
        'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'META', 'NVDA', 'INTC', 'CSCO', 'ADBE', 'CRM', 'ORCL', 'IBM', 'AMD', 'TSM',
        # Financial
        'JPM', 'BAC', 'GS', 'V', 'MA', 'BRK.B', 'C', 'MS', 'AXP', 'BLK', 'WFC', 'PNC', 'TFC', 'USB', 'COF',
        # Healthcare
        'JNJ', 'PFE', 'MRK', 'UNH', 'ABT', 'TMO', 'BMY', 'ABBV', 'LLY', 'AMGN', 'MDT', 'ISRG', 'CVS', 'GILD', 'BIIB',
        # Consumer
        'PG', 'KO', 'PEP', 'WMT', 'COST', 'MCD', 'NKE', 'SBUX', 'HD', 'TGT', 'LOW', 'DIS', 'NFLX', 'AMZN', 'EBAY',
        # Industrial
        'BA', 'GE', 'CAT', 'MMM', 'HON', 'UPS', 'LMT', 'RTX', 'DE', 'FDX', 'UNP', 'EMR', 'ETN', 'ITW', 'CMI',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'MPC', 'OXY', 'KMI', 'VLO',
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'WEC', 'ES', 'PEG',
        # Communication Services
        'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS', 'ATVI', 'EA', 'TTWO', 'OMC', 'IPG',
        # Real Estate
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'AVB', 'ESS', 'DLR', 'SPG',
        # Materials
        'LIN', 'APD', 'ECL', 'SHW', 'NUE', 'FCX', 'NEM', 'DOW', 'DD', 'PPG'
    ]
    
    # Generate additional tickers if needed
    if num_stocks > len(real_tickers):
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        additional_tickers = []
        
        while len(real_tickers) + len(additional_tickers) < num_stocks:
            new_ticker = ''.join(random.choices(letters, k=random.randint(3, 5)))
            if new_ticker not in real_tickers and new_ticker not in additional_tickers:
                additional_tickers.append(new_ticker)
        
        tickers = real_tickers + additional_tickers
        # Limit to the requested number of stocks
        tickers = tickers[:num_stocks]
    else:
        tickers = real_tickers[:num_stocks]
    
    # Define comprehensive sectors and industries with accurate groupings
    sectors_industries = {
        'Technology': [
            'Software', 'Hardware', 'Semiconductors', 'Internet', 'IT Services', 
            'Cloud Computing', 'Consumer Electronics', 'Enterprise Software', 'Cybersecurity'
        ],
        'Financial Services': [
            'Banks', 'Insurance', 'Asset Management', 'Payment Processing', 'Investment Banking',
            'Wealth Management', 'Consumer Finance', 'Mortgage Finance', 'Financial Data', 'Exchanges'
        ],
        'Healthcare': [
            'Pharmaceuticals', 'Medical Devices', 'Biotechnology', 'Healthcare Services', 'Health Insurance',
            'Life Sciences Tools', 'Hospitals', 'Managed Healthcare', 'Healthcare Technology', 'Medical Distribution'
        ],
        'Consumer Cyclical': [
            'Retail', 'Apparel', 'Restaurants', 'Entertainment', 'Automotive',
            'Homebuilding', 'Hotels', 'Luxury Goods', 'E-Commerce', 'Travel'
        ],
        'Consumer Defensive': [
            'Consumer Packaged Goods', 'Food & Beverage', 'Discount Stores', 'Grocery', 'Household Products',
            'Tobacco', 'Personal Products', 'Drug Stores', 'Agricultural Products', 'Food Distribution'
        ],
        'Industrial': [
            'Aerospace & Defense', 'Machinery', 'Transportation', 'Building Products', 'Electrical Equipment',
            'Industrial Distribution', 'Waste Management', 'Construction', 'Business Services', 'Engineering'
        ],
        'Energy': [
            'Oil & Gas E&P', 'Oil & Gas Integrated', 'Oil & Gas Refining', 'Oil & Gas Midstream', 'Oil & Gas Services',
            'Renewable Energy', 'Coal', 'Gas Utilities', 'Alternative Energy', 'Energy Equipment'
        ],
        'Utilities': [
            'Electric Utilities', 'Multi-Utilities', 'Water Utilities', 'Independent Power Producers', 'Renewable Utilities'
        ],
        'Communication Services': [
            'Telecom', 'Media', 'Entertainment', 'Social Media', 'Advertising',
            'Publishing', 'Broadcasting', 'Gaming', 'Wireless Communication', 'Cable & Satellite'
        ],
        'Real Estate': [
            'REITs', 'Property Management', 'Real Estate Development', 'Real Estate Services', 'Storage REITs',
            'Residential REITs', 'Office REITs', 'Industrial REITs', 'Retail REITs', 'Healthcare REITs'
        ],
        'Materials': [
            'Chemicals', 'Metals & Mining', 'Construction Materials', 'Containers & Packaging', 'Paper & Forest Products',
            'Steel', 'Specialty Chemicals', 'Gold', 'Agricultural Chemicals', 'Diversified Metals'
        ]
    }
    
    # Create realistic sector distribution - based on market cap weighting
    sector_weights = {
        'Technology': 0.25,
        'Financial Services': 0.15, 
        'Healthcare': 0.15,
        'Consumer Cyclical': 0.10,
        'Consumer Defensive': 0.08,
        'Industrial': 0.08,
        'Energy': 0.05,
        'Utilities': 0.03,
        'Communication Services': 0.04,
        'Real Estate': 0.03,
        'Materials': 0.04
    }
    
    # Create assignments based on weights
    sectors = []
    for sector, weight in sector_weights.items():
        count = int(weight * num_stocks)
        sectors.extend([sector] * count)
    
    # Adjust for any rounding errors
    while len(sectors) < num_stocks:
        sectors.append(random.choice(list(sector_weights.keys())))
    
    # Shuffle sectors
    random.shuffle(sectors)
    
    # Assign sector and industry to each ticker
    ticker_sectors = {}
    ticker_industries = {}
    
    for i, ticker in enumerate(tickers):
        sector = sectors[i]
        industry = random.choice(sectors_industries[sector])
        ticker_sectors[ticker] = sector
        ticker_industries[ticker] = industry
    
    # Generate realistic financial data with correlations between metrics
    data = []
    for ticker in tickers:
        # Basic info
        sector = ticker_sectors[ticker]
        industry = ticker_industries[ticker]
        
        # Base metrics with realistic relationships
        # Price and market cap are correlated with sector
        sector_multiplier = {
            'Technology': random.uniform(1.2, 2.0),
            'Financial Services': random.uniform(0.8, 1.2),
            'Healthcare': random.uniform(1.0, 1.5),
            'Consumer Cyclical': random.uniform(0.7, 1.3),
            'Consumer Defensive': random.uniform(0.9, 1.2),
            'Industrial': random.uniform(0.8, 1.1),
            'Energy': random.uniform(0.6, 1.0),
            'Utilities': random.uniform(0.5, 0.9),
            'Communication Services': random.uniform(0.7, 1.2),
            'Real Estate': random.uniform(0.6, 1.0),
            'Materials': random.uniform(0.5, 0.9)
        }[sector]
        
        # Base financial variables with realistic constraints
        base_growth = random.uniform(-5, 25) * sector_multiplier
        base_profitability = random.uniform(5, 30) * sector_multiplier
        base_risk = random.uniform(0.5, 2.0)
        
        # Financial metrics
        price = round(random.uniform(20, 500) * sector_multiplier, 2)
        
        # PE ratio depends on growth and sector
        if base_growth < 0:  # Negative growth often leads to high or negative P/E
            pe_ratio = round(random.uniform(20, 50) if random.random() > 0.3 else random.uniform(-50, -10), 2)
        elif base_growth > 15:  # High growth tends to have higher P/E
            pe_ratio = round(random.uniform(25, 100), 2)
        else:  # Moderate growth
            pe_ratio = round(random.uniform(10, 40), 2)
        
        # Some stocks might not have P/E (startups, biotech, etc.)
        if random.random() < 0.05 or sector == 'Healthcare' and industry == 'Biotechnology' and random.random() < 0.5:
            pe_ratio = float('nan')
        
        # Market cap is based on price and sector
        # More established sectors tend to have higher market caps
        base_shares = random.randint(100, 10000) * 1000000  # Base shares outstanding between 100M and 10B
        shares_outstanding = int(base_shares * sector_multiplier)
        market_cap = price * shares_outstanding / 1000000000  # In billions
        
        # Dividend yield based on sector and growth
        # Fast-growing companies often have low or no dividends
        # Stable sectors like utilities often have higher dividends
        if sector in ['Utilities', 'Consumer Defensive', 'Energy', 'Real Estate'] or base_growth < 5:
            dividend_yield = round(random.uniform(1, 5), 2)
        elif sector in ['Financial Services', 'Materials', 'Industrial'] and base_growth < 10:
            dividend_yield = round(random.uniform(0.5, 3), 2)
        elif random.random() < 0.3:  # Some companies in any sector might pay dividends
            dividend_yield = round(random.uniform(0.1, 2), 2)
        else:
            dividend_yield = 0
        
        # Beta (volatility) - related to sector
        if sector in ['Technology', 'Consumer Cyclical', 'Energy']:
            beta = round(random.uniform(1.0, 2.5), 2)
        elif sector in ['Healthcare', 'Financial Services', 'Communication Services']:
            beta = round(random.uniform(0.8, 1.8), 2)
        else:
            beta = round(random.uniform(0.5, 1.2), 2)
        
        # Risk adjustment based on company size
        if market_cap < 2:  # Small cap adjustment
            beta *= random.uniform(1.1, 1.3)
        elif market_cap > 100:  # Very large cap adjustment
            beta *= random.uniform(0.8, 0.95)
        
        # Growth metrics - PEG relates PE to growth rate
        earnings_growth_3yr = round(base_growth + random.uniform(-5, 5), 2)
        revenue_growth_3yr = round(earnings_growth_3yr + random.uniform(-3, 3), 2)
        
        # PEG ratio - Price/Earnings to Growth
        if not pd.isna(pe_ratio) and earnings_growth_3yr > 0:
            peg_ratio = round(pe_ratio / earnings_growth_3yr, 2)
            # Very high PEG usually indicates overvaluation or error, cap it
            peg_ratio = min(5.0, peg_ratio)
        else:
            peg_ratio = float('nan')
        
        # Profitability metrics - related to sector and growth
        profit_margin = round(base_profitability + random.uniform(-5, 5), 2)
        operating_margin = round(profit_margin + random.uniform(-3, 8), 2)
        
        # Adjust profitability for specific industries
        if industry in ['Biotechnology', 'Pharmaceuticals'] and random.random() < 0.4:
            # Many biotech companies are pre-profit
            profit_margin = round(random.uniform(-50, 0), 2)
            operating_margin = round(random.uniform(-60, -10), 2)
        
        # Return metrics based on profitability
        roa = round(profit_margin * random.uniform(0.2, 0.6), 2)  # ROA is usually lower than profit margin
        roe = round(profit_margin * random.uniform(1.0, 2.5), 2)  # ROE is usually higher than profit margin
        
        # Valuation metrics
        pb_ratio = round(random.uniform(1, 8) * (1 + (earnings_growth_3yr/100)), 2)
        ps_ratio = round(random.uniform(1, 10) * (1 + (revenue_growth_3yr/100)), 2)
        ev_to_ebitda = round(random.uniform(5, 25) * (1 + (earnings_growth_3yr/200)), 2)
        
        # Debt metrics - related to sector stability
        if sector in ['Utilities', 'Real Estate', 'Financial Services']:
            debt_to_equity = round(random.uniform(1.0, 3.0), 2)
        else:
            debt_to_equity = round(random.uniform(0.1, 1.5), 2)
        
        # Current ratio (liquidity)
        if sector in ['Technology', 'Healthcare'] and profit_margin > 15:
            current_ratio = round(random.uniform(1.5, 4.0), 2)  # Tech often has better liquidity
        else:
            current_ratio = round(random.uniform(0.8, 2.5), 2)
        
        # Historical performance
        # Stock returns are correlated with growth and overall market
        market_performance = random.uniform(-5, 20)  # General market performance
        stock_alpha = (earnings_growth_3yr - 10) / 5  # Company-specific performance
        avg_return_1yr = round(market_performance + stock_alpha + random.uniform(-15, 15), 2)
        
        # Volatility is related to beta and size
        volatility_1yr = round(15 * beta * (1 + random.uniform(-0.3, 0.3)), 2)
        
        # Risk-adjusted metrics
        risk_free_rate = 2.0  # Assume 2% risk-free rate
        if volatility_1yr > 0:
            sharpe_ratio = round((avg_return_1yr - risk_free_rate) / volatility_1yr, 2)
        else:
            sharpe_ratio = 0
        
        max_drawdown = round(-1 * (volatility_1yr * random.uniform(1.5, 3.0)), 2)
        
        # ESG metrics - some sectors score better than others
        if sector in ['Technology', 'Healthcare', 'Utilities']:
            esg_base = random.uniform(60, 90)
        elif sector in ['Financial Services', 'Consumer Defensive', 'Communication Services']:
            esg_base = random.uniform(50, 80)
        else:
            esg_base = random.uniform(30, 70)
        
        esg_score = round(esg_base + random.uniform(-10, 10), 2)
        esg_score = max(min(esg_score, 100), 0)  # Clamp between 0 and 100
        
        # Market type based on market cap
        if market_cap > 50:
            market_type = 'Large Cap'
        elif market_cap > 10:
            market_type = 'Mid Cap'
        else:
            market_type = 'Small Cap'
        
        # Exchange distribution
        if random.random() < 0.6:  # 60% on NYSE
            exchange = 'NYSE'
        else:
            exchange = 'NASDAQ'
        
        # Create company name - often related to ticker
        company_name = f"{ticker} {random.choice(['Inc.', 'Corp.', 'Group', 'Technologies', 'Holdings', 'Pharmaceuticals', 'Energy', 'Solutions', 'Systems', 'Brands'])}"
        
        data.append({
            'ticker': ticker,
            'company_name': company_name,
            'sector': sector,
            'industry': industry,
            'market_type': market_type,
            'exchange': exchange,
            'price': price,
            'market_cap': market_cap,
            'pe_ratio': pe_ratio,
            'peg_ratio': peg_ratio,
            'pb_ratio': pb_ratio,
            'ps_ratio': ps_ratio,
            'dividend_yield': dividend_yield,
            'beta': beta,
            'profit_margin': profit_margin,
            'operating_margin': operating_margin,
            'roa': roa,
            'roe': roe,
            'ev_to_ebitda': ev_to_ebitda,
            'debt_to_equity': debt_to_equity,
            'current_ratio': current_ratio,
            'revenue_growth_3yr': revenue_growth_3yr,
            'earnings_growth_3yr': earnings_growth_3yr,
            'shares_outstanding': shares_outstanding,
            'avg_return_1yr': avg_return_1yr,
            'volatility_1yr': volatility_1yr,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'esg_score': esg_score
        })
    
    # Create DataFrame
    stocks_df = pd.DataFrame(data)
    
    # ---- Generate historical price data ----
    def generate_historical_prices(ticker_data, days=252*2, end_date=None):
        """Generate 2 years of historical price data with realistic patterns"""
        ticker = ticker_data['ticker']
        current_price = ticker_data['price']
        volatility = ticker_data['volatility_1yr'] / 100
        beta = ticker_data['beta']
        growth_trend = ticker_data['revenue_growth_3yr'] / 100 / 252  # Daily growth factor
        
        # Adjusting for market trend
        market_trend = 0.0001  # Slight upward bias in market
        
        # Base parameters
        daily_volatility = volatility / np.sqrt(252)
        
        if not end_date:
            end_date = datetime.now().date()
        
        start_date = end_date - timedelta(days=days)
        
        # Generate dates (trading days only)
        all_dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday to Friday
                all_dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Generate market returns (common factor)
        market_returns = np.random.normal(market_trend, 0.01, len(all_dates))
        
        # Add seasonality and some market cycles (simplified)
        t = np.arange(len(all_dates))
        # Add a slow cycle (approx 1 year)
        market_cycle = 0.0002 * np.sin(2 * np.pi * t / 252)
        # Add a faster cycle (approx 1 quarter)
        market_cycle += 0.0001 * np.sin(2 * np.pi * t / 63)
        market_returns += market_cycle
        
        # Generate stock-specific returns
        stock_specific_vol = daily_volatility * (1 - 0.6)  # Assuming 60% of volatility is market-related
        stock_specific = np.random.normal(growth_trend, stock_specific_vol, len(all_dates))
        
        # Combine returns (beta * market + stock specific)
        total_returns = (beta * market_returns) + stock_specific
        
        # Convert returns to prices
        prices = [current_price / (1 + sum(total_returns[i:]) + 0.0001) for i in range(len(total_returns))]
        prices.reverse()  # Since we calculated backward from current price
        
        # Create DataFrame
        price_data = pd.DataFrame({
            'date': all_dates,
            'ticker': ticker,
            'price': prices
        })
        
        return price_data
    
    # Generate prices for all stocks (this can be time-consuming for many stocks)
    # To save time, let's generate for a subset of stocks
    price_sample_size = min(50, num_stocks)  # Generate for at most 50 stocks
    historical_prices = pd.DataFrame()
    
    for i, row in stocks_df.head(price_sample_size).iterrows():
        ticker_prices = generate_historical_prices(row)
        historical_prices = pd.concat([historical_prices, ticker_prices])
    
    # ---- Generate user portfolios ----
    def generate_user_portfolios(stocks_df, num_users=1000):
        tickers = stocks_df['ticker'].tolist()
        sectors = {ticker: sector for ticker, sector in zip(stocks_df['ticker'], stocks_df['sector'])}
        
        user_data = []
        for user_id in range(1, num_users+1):
            # Different user investment strategies
            strategy = random.choice([
                'diversified',       # Holds stocks across many sectors
                'sector_focused',    # Focuses on 1-2 sectors
                'large_cap_biased',  # Prefers large caps
                'small_cap_biased',  # Prefers small caps
                'dividend_focused',  # Focuses on dividend stocks
                'growth_focused',    # Focuses on growth stocks
                'random'             # Random selection
            ])
            
            # Different portfolio sizes based on strategy
            if strategy == 'diversified':
                portfolio_size = random.randint(8, 20)
            elif strategy == 'sector_focused':
                portfolio_size = random.randint(5, 12)
            else:
                portfolio_size = random.randint(3, 15)
                
            # Select stocks based on strategy
            if strategy == 'sector_focused':
                # Choose 1-2 sectors to focus on
                focus_sectors = random.sample(list(set(sectors.values())), random.randint(1, 2))
                candidate_tickers = [t for t, s in sectors.items() if s in focus_sectors]
                # Ensure we have enough stocks
                if len(candidate_tickers) < portfolio_size:
                    # Add some stocks from other sectors
                    other_tickers = [t for t in tickers if t not in candidate_tickers]
                    candidate_tickers.extend(random.sample(other_tickers, portfolio_size - len(candidate_tickers)))
                selected_tickers = random.sample(candidate_tickers, min(portfolio_size, len(candidate_tickers)))
            
            elif strategy == 'large_cap_biased':
                # Prefer stocks with larger market caps
                large_caps = stocks_df[stocks_df['market_type'] == 'Large Cap']['ticker'].tolist()
                if len(large_caps) >= portfolio_size * 0.7:
                    primary = random.sample(large_caps, int(portfolio_size * 0.7))
                    secondary = random.sample([t for t in tickers if t not in primary], portfolio_size - len(primary))
                    selected_tickers = primary + secondary
                else:
                    selected_tickers = random.sample(tickers, portfolio_size)
            
            elif strategy == 'small_cap_biased':
                # Prefer stocks with smaller market caps
                small_caps = stocks_df[stocks_df['market_type'] == 'Small Cap']['ticker'].tolist()
                if len(small_caps) >= portfolio_size * 0.7:
                    primary = random.sample(small_caps, int(portfolio_size * 0.7))
                    secondary = random.sample([t for t in tickers if t not in primary], portfolio_size - len(primary))
                    selected_tickers = primary + secondary
                else:
                    selected_tickers = random.sample(tickers, portfolio_size)
            
            elif strategy == 'dividend_focused':
                # Prefer stocks with dividends
                dividend_stocks = stocks_df[stocks_df['dividend_yield'] > 0]['ticker'].tolist()
                if len(dividend_stocks) >= portfolio_size * 0.8:
                    primary = random.sample(dividend_stocks, int(portfolio_size * 0.8))
                    secondary = random.sample([t for t in tickers if t not in primary], portfolio_size - len(primary))
                    selected_tickers = primary + secondary
                else:
                    selected_tickers = random.sample(tickers, portfolio_size)
            
            elif strategy == 'growth_focused':
                # Prefer growth stocks
                growth_stocks = stocks_df[stocks_df['earnings_growth_3yr'] > 10]['ticker'].tolist()
                if len(growth_stocks) >= portfolio_size * 0.7:
                    primary = random.sample(growth_stocks, int(portfolio_size * 0.7))
                    secondary = random.sample([t for t in tickers if t not in primary], portfolio_size - len(primary))
                    selected_tickers = primary + secondary
                else:
                    selected_tickers = random.sample(tickers, portfolio_size)
            
            else:  # diversified or random
                selected_tickers = random.sample(tickers, portfolio_size)
            
            # Generate portfolio weights that sum to 1
            # Use a Dirichlet distribution for more realistic allocations
            alpha = [1.0] * len(selected_tickers)  # Equal concentration parameter
            
            # Some investors heavily weight certain positions
            if random.random() < 0.3:  # 30% of users have concentrated positions
                # Randomly boost 1-3 positions
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, len(alpha)-1)
                    alpha[idx] = random.uniform(3.0, 10.0)  # Higher alpha = higher weight
            
            weights = np.random.dirichlet(alpha)
            weights = weights / weights.sum()  # Normalize to ensure sum = 1
            
            # Record user portfolio
            for i, ticker in enumerate(selected_tickers):
                user_data.append({
                    'user_id': f'user_{user_id}',
                    'ticker': ticker,
                    'weight': round(weights[i], 4)
                })
                
            # Add some user interactions/views (for collaborative filtering)
            # Users often view stocks similar to what they own
            viewed_tickers = set()
            
            # First add some views of owned stocks (users research what they own)
            for ticker in selected_tickers:
                if random.random() < 0.7:  # 70% chance they've viewed each owned stock
                    viewed_tickers.add(ticker)
            
            # Then add other viewed stocks
            candidate_tickers = [t for t in tickers if t not in viewed_tickers]
            # Calculate how many additional views we can add
            remaining_views = min(15 - len(viewed_tickers), len(candidate_tickers))
            if remaining_views > 0:
                additional_views = random.sample(candidate_tickers, remaining_views)
                viewed_tickers.update(additional_views)
            
            # Create view interactions with different engagement levels
            for ticker in viewed_tickers:
                engagement = random.choice(['view', 'research', 'watchlist', 'comparison'])
                frequency = random.randint(1, 10)  # How many times they interacted
                
                user_data.append({
                    'user_id': f'user_{user_id}',
                    'ticker': ticker,
                    'interaction_type': engagement,
                    'interaction_count': frequency
                })
        
        return pd.DataFrame(user_data)
    
    # Generate user portfolios with a meaningful number of users
    num_users = min(5000, num_stocks * 10)  # Scale users with stock count
    user_data = generate_user_portfolios(stocks_df, num_users)
    
    # Split user data into portfolios and interactions
    user_portfolios = user_data[user_data['weight'].notna()].copy()
    user_interactions = user_data[user_data['interaction_type'].notna()].copy()
    
    # Save datasets
    stocks_data_path = os.path.join(output_dir, 'stocks_data.csv')
    stocks_df.to_csv(stocks_data_path, index=False)
    
    historical_prices_path = os.path.join(output_dir, 'historical_prices.csv')
    historical_prices.to_csv(historical_prices_path, index=False)
    
    user_portfolios_path = os.path.join(output_dir, 'user_portfolios.csv')
    user_portfolios.to_csv(user_portfolios_path, index=False)
    
    user_interactions_path = os.path.join(output_dir, 'user_interactions.csv')
    user_interactions.to_csv(user_interactions_path, index=False)
    
    print(f"Generated {len(stocks_df)} stocks")
    print(f"Generated {historical_prices['ticker'].nunique()} stocks with historical price data")
    print(f"Generated {user_portfolios['user_id'].nunique()} user portfolios")
    print(f"Generated {len(user_interactions)} user interactions")
    
    return stocks_data_path, historical_prices_path, user_portfolios_path, user_interactions_path

# Generate datasets
print("Starting data generation...")
generate_comprehensive_stock_data(num_stocks=500, output_dir="stock_recommender_data")

# Print samples to verify
print("\nTrying to read generated files...")
try:
    print("\nReading stocks data...")
    stocks_df = pd.read_csv("stock_recommender_data/stocks_data.csv")
    print("Successfully read stocks data")
    print("\nSample stock data:")
    print(stocks_df.head())

    print("\nReading historical price data...")
    historical_prices = pd.read_csv("stock_recommender_data/historical_prices.csv")
    print("Successfully read historical prices")
    print("\nSample historical price data:")
    print(historical_prices.head())

    print("\nReading user portfolios...")
    user_portfolios = pd.read_csv("stock_recommender_data/user_portfolios.csv")
    print("Successfully read user portfolios")
    print("\nSample user portfolios:")
    print(user_portfolios.head())

    print("\nReading user interactions...")
    user_interactions = pd.read_csv("stock_recommender_data/user_interactions.csv")
    print("Successfully read user interactions")
    print("\nSample user interactions:")
    print(user_interactions.head())

    # Now let's verify the size of our datasets
    print("\nDataset sizes:")
    print(f"Stocks: {len(stocks_df)} records")
    print(f"Historical prices: {len(historical_prices)} records")
    print(f"User portfolios: {len(user_portfolios)} records")
    print(f"User interactions: {len(user_interactions)} records")
except Exception as e:
    print(f"Error reading files: {str(e)}")

print("\nData generation complete. You can now use these files to train your stock recommendation system.")

if __name__ == "__main__":
    generate_comprehensive_stock_data(num_stocks=500, output_dir="stock_recommender_data")
