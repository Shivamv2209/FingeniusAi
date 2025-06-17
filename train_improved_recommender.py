import os
from improved_recommender import ImprovedStockRecommender
import pandas as pd
import textwrap

def format_explanation(explanation, width=80):
    """Format explanation text with proper line wrapping."""
    return "\n      ".join(textwrap.wrap(explanation, width=width))

def main():
    # Initialize paths
    data_dir = "stock_recommender_data"
    stocks_data_path = os.path.join(data_dir, "stocks_data.csv")
    unique_portfolios_path = os.path.join(data_dir, "users_unique_portfolio.csv")
    model_path = "improved_stock_recommender.pkl"
    
    print("Initializing improved recommender system...")
    recommender = ImprovedStockRecommender()
    
    # Load data
    print(f"Loading stock data from {stocks_data_path}")
    print(f"Loading unique portfolios from {unique_portfolios_path}")
    recommender.load_data(
        stocks_data_path=stocks_data_path,
        unique_portfolios_path=unique_portfolios_path
    )
    
    # Prepare features
    print("Preparing features...")
    recommender.prepare_features()
    
    # Save model
    print(f"Saving trained model to {model_path}")
    recommender.save_model(model_path)
    
    # Demonstrate new features
    
    # 1. Portfolio risk analysis for a sample user
    sample_user = "user_500"  # A user with technology concentration
    print(f"\n--- Portfolio Risk Analysis for {sample_user} ---\n")
    
    try:
        # Get portfolio composition
        user_portfolio = recommender.expand_user_portfolio(sample_user)
        
        # Calculate sector distribution
        sector_weights = {}
        total_weight = 0
        
        for _, row in user_portfolio.iterrows():
            ticker = row['ticker']
            weight = row['weight']
            total_weight += weight
            
            if ticker in recommender.stocks_data.index:
                sector = recommender.stocks_data.loc[ticker]['sector']
                if sector in sector_weights:
                    sector_weights[sector] += weight
                else:
                    sector_weights[sector] = weight
        
        # Normalize weights
        for sector in sector_weights:
            sector_weights[sector] = sector_weights[sector] / total_weight if total_weight > 0 else 0
        
        # Sort sectors by weight
        sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
        
        print("Sector Distribution:")
        for sector, weight in sorted_sectors:
            print(f"- {sector}: {weight*100:.1f}%")
        
        # Risk analysis
        risk_analysis = recommender.analyze_portfolio_risks(sample_user)
        
        print("\nRisk Assessment:")
        if risk_analysis['alerts']:
            print("⚠️ ALERTS:")
            for alert in risk_analysis['alerts']:
                print(f"- {alert['message']}")
        else:
            print("✓ No major concentration risks detected.")
        
        # 2. Demonstrate diversification recommendations
        if any(alert['type'] == 'sector_overconcentration' for alert in risk_analysis['alerts']):
            print("\n--- Diversification Recommendations ---")
            
            diversification_recs = recommender.generate_diversification_recommendations(sample_user, n=3)
            
            if diversification_recs:
                print(f"\nRecommended stocks for diversification:")
                for i, stock in enumerate(diversification_recs):
                    print(f"{i+1}. {stock['ticker']} ({stock['company_name']})")
                    print(f"   Sector: {stock['sector']} (New sector)")
                    print(f"   Similarity: {stock['similarity_score']:.3f}")
                    print(f"   Explanation: {format_explanation(stock['explanation'])}")
                    print()
        
        # 3. Standard recommendations with explanations
        print("\n--- Standard Recommendations with Explanations ---")
        
        recommendations = recommender.generate_recommendations(sample_user, n=3, include_explanations=True)
        
        print(f"\nRecommended stocks based on preference:")
        for i, stock in enumerate(recommendations):
            print(f"{i+1}. {stock['ticker']} ({stock['company_name']})")
            print(f"   Sector: {stock['sector']}")
            print(f"   Similarity: {stock['similarity_score']:.3f}")
            print(f"   Explanation: {format_explanation(stock['explanation'])}")
            print()
            
    except Exception as e:
        print(f"Error demonstrating features: {str(e)}")
    
    print("\nModel training and feature demonstration complete!")

if __name__ == "__main__":
    main() 