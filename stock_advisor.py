#!/usr/bin/env python
import argparse
import os
import sys
import textwrap
from improved_recommender import ImprovedStockRecommender
import pandas as pd
import json

def format_explanation(explanation, width=70):
    """Format explanation text with proper line wrapping."""
    return "\n      ".join(textwrap.wrap(explanation, width=width))

class StockAdvisor:
    def __init__(self):
        self.model_path = "improved_stock_recommender.pkl"
        self.recommender = None
    
    def load_model(self):
        """Load the trained model or train if not available."""
        if os.path.exists(self.model_path):
            print(f"Loading recommender model from {self.model_path}...")
            self.recommender = ImprovedStockRecommender()
            self.recommender.load_model(self.model_path)
        else:
            print(f"Model not found at {self.model_path}. Training a new model...")
            self.train_model()
        
        return self.recommender
    
    def train_model(self):
        """Train the recommender model with the available data."""
        data_dir = "stock_recommender_data"
        stocks_data_path = os.path.join(data_dir, "stocks_data.csv")
        unique_portfolios_path = os.path.join(data_dir, "users_unique_portfolio.csv")
        
        if not os.path.exists(stocks_data_path):
            print(f"Error: Stock data not found at {stocks_data_path}")
            sys.exit(1)
            
        if not os.path.exists(unique_portfolios_path):
            print(f"Error: User portfolio data not found at {unique_portfolios_path}")
            sys.exit(1)
        
        print("Initializing recommender system...")
        self.recommender = ImprovedStockRecommender()
        
        print(f"Loading stock data from {stocks_data_path}")
        print(f"Loading user portfolios from {unique_portfolios_path}")
        
        self.recommender.load_data(
            stocks_data_path=stocks_data_path,
            unique_portfolios_path=unique_portfolios_path
        )
        
        print("Preparing features...")
        self.recommender.prepare_features()
        
        print(f"Saving trained model to {self.model_path}")
        self.recommender.save_model(self.model_path)
        
        return self.recommender
    
    def get_portfolio_report(self, user_id):
        """Generate a streamlined portfolio report with only alerts and recommendations for a user."""
        if not self.recommender:
            self.load_model()
        
        try:
            # Get the portfolio
            user_portfolio = self.recommender.expand_user_portfolio(user_id)
            
            print(f"\n===== QUICK PORTFOLIO ANALYSIS FOR {user_id} =====\n")
            
            # Risk analysis - only show alerts if present
            print("RISK ASSESSMENT:")
            print("---------------")
            try:
                risk_analysis = self.recommender.analyze_portfolio_risks(user_id)
                
                if risk_analysis['alerts']:
                    print("⚠️ ALERTS:")
                    for alert in risk_analysis['alerts']:
                        print(f"- {alert['message']}")
                    print()
                else:
                    print("✓ No major concentration risks detected in your portfolio.\n")
            except Exception as e:
                print(f"Could not analyze portfolio risks: {str(e)}\n")
            
            # Generate recommendations
            print("RECOMMENDED STOCKS:")
            print("------------------")
            try:
                recommendations = self.recommender.generate_recommendations(user_id, n=5, include_explanations=True)
                
                for i, stock in enumerate(recommendations):
                    # Make sure all fields are properly converted
                    company_name = stock['company_name']
                    if isinstance(company_name, pd.Series):
                        company_name = str(company_name.iloc[0]) if not company_name.empty else "Unknown"
                    
                    sector = stock['sector']
                    if isinstance(sector, pd.Series):
                        sector = str(sector.iloc[0]) if not sector.empty else "Unknown"
                    
                    price = stock['price']
                    if isinstance(price, pd.Series):
                        price = float(price.iloc[0]) if not price.empty else 0.0
                    
                    print(f"{i+1}. {stock['ticker']} ({company_name})")
                    print(f"   Sector: {sector}")
                    print(f"   Price: ${float(price):.2f}")
                    if 'explanation' in stock:
                        print(f"   Why: {format_explanation(stock['explanation'])}")
                    print()
            except Exception as e:
                print(f"Error generating recommendations: {str(e)}")
            
            # If there are sector concentration issues, show diversification recommendations
            try:
                risk_analysis = self.recommender.analyze_portfolio_risks(user_id)
                has_concentration = False
                
                if risk_analysis and 'alerts' in risk_analysis:
                    for alert in risk_analysis['alerts']:
                        # Make sure alert is a dict and has a 'type' key
                        if isinstance(alert, dict) and 'type' in alert:
                            alert_type = alert['type']
                            # Convert to string if needed
                            if isinstance(alert_type, pd.Series):
                                alert_type = str(alert_type.iloc[0]) if not alert_type.empty else ""
                            
                            if alert_type == 'sector_overconcentration':
                                has_concentration = True
                                break
                
                if has_concentration:
                    print("DIVERSIFICATION RECOMMENDATIONS:")
                    print("------------------------------")
                    print("Consider adding these stocks from sectors not in your portfolio:\n")
                    
                    try:
                        diversification_recs = self.recommender.generate_diversification_recommendations(user_id, n=3)
                        
                        if not diversification_recs:
                            print("Could not find suitable diversification recommendations.")
                        else:
                            for i, stock in enumerate(diversification_recs):
                                # Make sure all fields are properly converted
                                company_name = stock['company_name']
                                if isinstance(company_name, pd.Series):
                                    company_name = str(company_name.iloc[0]) if not company_name.empty else "Unknown"
                                
                                sector = stock['sector']
                                if isinstance(sector, pd.Series):
                                    sector = str(sector.iloc[0]) if not sector.empty else "Unknown"
                                
                                price = stock['price']
                                if isinstance(price, pd.Series):
                                    price = float(price.iloc[0]) if not price.empty else 0.0
                                
                                print(f"{i+1}. {stock['ticker']} ({company_name})")
                                print(f"   Sector: {sector} (New sector for your portfolio)")
                                print(f"   Price: ${float(price):.2f}")
                                if 'explanation' in stock:
                                    print(f"   Why: {format_explanation(stock['explanation'])}")
                                print()
                    except Exception as e:
                        print(f"Error generating diversification recommendations: {str(e)}")
            except Exception as e:
                print(f"Error checking for sector concentration: {str(e)}")
            
            return True
        
        except Exception as e:
            print(f"Error analyzing portfolio: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def explain_stock(self, ticker):
        """Provide a simple explanation of a stock."""
        if not self.recommender:
            self.load_model()
        
        try:
            if ticker not in self.recommender.stocks_data.index:
                print(f"Error: Ticker '{ticker}' not found in the dataset.")
                return False
                
            stock_info = self.recommender.stocks_data.loc[ticker].to_dict()
            
            print(f"\n===== STOCK EXPLANATION: {ticker} =====\n")
            print(f"Company Name: {stock_info['company_name']}")
            print(f"Sector: {stock_info['sector']}")
            
            # Price assessment
            price = stock_info['price']
            if isinstance(price, (int, float)):
                print(f"Price: ${float(price):.2f}")
                
                # Context about price
                price_range = self.recommender.stocks_data['price'].quantile([0.25, 0.5, 0.75]).to_dict()
                if price < price_range[0.25]:
                    print("This is in the lower 25% of stock prices in our database.")
                elif price > price_range[0.75]:
                    print("This is in the upper 25% of stock prices in our database.")
                else:
                    print("This is in the middle range of stock prices in our database.")
            
            # Company size
            market_cap = stock_info['market_cap']
            if isinstance(market_cap, (int, float)):
                mc_val = float(market_cap)
                if mc_val > 10000:
                    size = "very large (mega-cap)"
                elif mc_val > 1000:
                    size = "large (large-cap)"
                elif mc_val > 100:
                    size = "medium-sized (mid-cap)"
                else:
                    size = "smaller (small-cap)"
                print(f"Company Size: {size} company")
            
            # ESG info
            esg_score = stock_info['esg_score']
            if isinstance(esg_score, (int, float)):
                esg_val = float(esg_score)
                if esg_val > 80:
                    print(f"ESG Score: {esg_val:.1f} - Excellent environmental and social practices")
                elif esg_val > 60:
                    print(f"ESG Score: {esg_val:.1f} - Good environmental and social practices")
                else:
                    print(f"ESG Score: {esg_val:.1f}")
            
            # Volatility/risk
            beta = stock_info['beta']
            if isinstance(beta, (int, float)):
                beta_val = float(beta)
                if beta_val > 1.5:
                    print(f"Volatility: High (Beta: {beta_val:.2f}) - The stock price tends to move more than the overall market")
                elif beta_val < 0.8:
                    print(f"Volatility: Low (Beta: {beta_val:.2f}) - The stock price tends to be more stable than the overall market")
                else:
                    print(f"Volatility: Medium (Beta: {beta_val:.2f}) - The stock price tends to move similar to the overall market")
            
            # Print similar stocks
            print("\nSIMILAR STOCKS:")
            print("--------------")
            similar = self.recommender.generate_recommendations(ticker, n=3, include_explanations=False)
            
            for i, stock in enumerate(similar):
                print(f"{i+1}. {stock['ticker']} ({stock['company_name']})")
                print(f"   Similarity: {stock['similarity_score']:.3f}")
                print(f"   Sector: {stock['sector']}")
                print()
            
            # Simple explanation
            print("SIMPLE EXPLANATION:")
            print("-----------------")
            if isinstance(beta, (int, float)) and isinstance(esg_score, (int, float)):
                explanation = f"{stock_info['company_name']} is a {size} company in the {stock_info['sector']} sector. "
                
                # Risk profile
                if beta_val > 1.5:
                    explanation += "It tends to be more volatile than the market, which means higher potential returns but also higher risk. "
                elif beta_val < 0.8:
                    explanation += "It tends to be more stable than the market, which means potentially lower risk. "
                    
                # ESG profile
                if esg_val > 80:
                    explanation += "The company has excellent environmental and social practices. "
                elif esg_val > 60:
                    explanation += "The company has good environmental and social practices. "
                
                print(format_explanation(explanation))
            
            return True
        
        except Exception as e:
            print(f"Error explaining stock: {str(e)}")
            return False
    
    def find_similar_stocks(self, ticker, count=5):
        """Find stocks similar to the provided ticker."""
        if not self.recommender:
            self.load_model()
        
        try:
            if ticker not in self.recommender.stocks_data.index:
                print(f"Error: Ticker '{ticker}' not found in the dataset.")
                return False
                
            print(f"\n===== STOCKS SIMILAR TO {ticker} =====\n")
            
            stock_info = self.recommender.stocks_data.loc[ticker].to_dict()
            print(f"Reference Stock: {ticker} ({stock_info['company_name']})")
            print(f"Sector: {stock_info['sector']}")
            print(f"Price: ${float(stock_info['price']):.2f}")
            print()
            
            similar = self.recommender.generate_recommendations(ticker, n=count, include_explanations=True)
            
            print(f"TOP {count} SIMILAR STOCKS:")
            print("--------------------")
            for i, stock in enumerate(similar):
                print(f"{i+1}. {stock['ticker']} ({stock['company_name']})")
                print(f"   Similarity Score: {stock['similarity_score']:.3f}")
                print(f"   Sector: {stock['sector']}")
                print(f"   Price: ${float(stock['price']):.2f}")
                if 'explanation' in stock:
                    print(f"   Why Similar: {format_explanation(stock['explanation'])}")
                print()
            
            return True
        
        except Exception as e:
            print(f"Error finding similar stocks: {str(e)}")
            return False
    
    def explore_sector(self, sector_name, count=5):
        """Explore stocks within a specific sector."""
        if not self.recommender:
            self.load_model()
        
        try:
            # Get unique sectors for validation
            available_sectors = set(self.recommender.stocks_data['sector'].unique())
            
            if sector_name not in available_sectors:
                print(f"Error: Sector '{sector_name}' not found. Available sectors are:")
                for sector in sorted(available_sectors):
                    print(f"- {sector}")
                return False
            
            # Filter stocks by sector
            sector_stocks = self.recommender.stocks_data[self.recommender.stocks_data['sector'] == sector_name]
            
            print(f"\n===== {sector_name.upper()} SECTOR ANALYSIS =====\n")
            print(f"Number of stocks in this sector: {len(sector_stocks)}")
            
            # Sort by market cap (largest first)
            sector_stocks = sector_stocks.sort_values('market_cap', ascending=False)
            
            print(f"\nTOP {min(count, len(sector_stocks))} STOCKS BY MARKET CAP:")
            print("-----------------------------")
            for i, (ticker, stock) in enumerate(sector_stocks.head(count).iterrows()):
                stock_dict = stock.to_dict()
                price = stock_dict['price']
                market_cap = stock_dict['market_cap']
                
                print(f"{i+1}. {ticker} ({stock_dict['company_name']})")
                print(f"   Market Cap: ${float(market_cap):.2f} billion")
                print(f"   Price: ${float(price):.2f}")
                if 'esg_score' in stock_dict:
                    print(f"   ESG Score: {float(stock_dict['esg_score']):.1f}")
                print()
            
            return True
        
        except Exception as e:
            print(f"Error exploring sector: {str(e)}")
            return False

def main():
    
    parser = argparse.ArgumentParser(
        description='Stock Advisor - A comprehensive stock recommendation and portfolio analysis tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
        Examples:
          python stock_advisor.py portfolio user_100    # Analyze portfolio for user_100
          python stock_advisor.py stock AAPL            # Explain stock AAPL
          python stock_advisor.py similar AAPL --count 3  # Find 3 stocks similar to AAPL
          python stock_advisor.py sector Technology     # Explore the Technology sector
          python stock_advisor.py train                 # Train or retrain the model
        ''')
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Command to run')
    
    # Portfolio analysis command
    portfolio_parser = subparsers.add_parser('portfolio', help='Analyze a user portfolio')
    portfolio_parser.add_argument('user_id', type=str, help='User ID')
    
    # Stock explanation command
    stock_parser = subparsers.add_parser('stock', help='Explain a specific stock')
    stock_parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    
    # Similar stocks command
    similar_parser = subparsers.add_parser('similar', help='Find similar stocks')
    similar_parser.add_argument('ticker', type=str, help='Reference stock ticker symbol')
    similar_parser.add_argument('--count', '-c', type=int, default=5, help='Number of similar stocks to find')
    
    # Sector exploration command
    sector_parser = subparsers.add_parser('sector', help='Explore stocks in a sector')
    sector_parser.add_argument('sector_name', type=str, help='Sector name')
    sector_parser.add_argument('--count', '-c', type=int, default=5, help='Number of top stocks to show')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train or retrain the model')
    
    args = parser.parse_args()
    
    # Initialize the advisor
    advisor = StockAdvisor()
    
    if args.command == 'portfolio':
        advisor.get_portfolio_report(args.user_id)
    elif args.command == 'stock':
        advisor.explain_stock(args.ticker)
    elif args.command == 'similar':
        advisor.find_similar_stocks(args.ticker, args.count)
    elif args.command == 'sector':
        advisor.explore_sector(args.sector_name, args.count)
    elif args.command == 'train':
        advisor.train_model()
        print("Training complete. Model is ready to use.")
        
    user_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not user_id:
        print(json.dumps({'error': 'User ID is required'}))
        return

    recommender = ImprovedStockRecommender()
    recommender.load_model("improved_stock_recommender.pkl")

    try:
        recommendations = recommender.generate_recommendations(user_id, n=3, include_explanations=True)
        print(json.dumps(recommendations))
    except Exception as e:
        print(json.dumps({'error': str(e)}))    

if __name__ == "__main__":
    main() 