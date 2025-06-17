import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class ImprovedStockRecommender:
    def __init__(self):
        self.stocks_data = None
        self.user_portfolios = None
        self.unique_portfolios = None
        self.stock_features = None
        self.feature_columns = [
            'price', 'market_cap', 'pe_ratio', 'peg_ratio', 'pb_ratio', 'ps_ratio',
            'dividend_yield', 'beta', 'profit_margin', 'operating_margin', 'roa', 'roe',
            'ev_to_ebitda', 'debt_to_equity', 'current_ratio', 'revenue_growth_3yr',
            'earnings_growth_3yr', 'avg_return_1yr', 'volatility_1yr', 'sharpe_ratio',
            'max_drawdown', 'esg_score'
        ]
        self.scaler = StandardScaler()
        # Define sector diversification thresholds
        self.sector_concentration_threshold = 0.5  # Alert if a sector is over 50%
        self.sector_count_min = 3  # Recommend having at least 3 sectors
        
    def load_data(self, stocks_data_path, user_portfolios_path=None, unique_portfolios_path=None):
        """
        Load stock data and user portfolios
        
        Parameters:
        stocks_data_path (str): Path to the CSV file containing stock features
        user_portfolios_path (str): Path to the CSV file containing standard user portfolios
        unique_portfolios_path (str): Path to the CSV file containing unique user portfolios format
        """
        # Load stock features
        self.stocks_data = pd.read_csv(stocks_data_path)
        self.stocks_data.set_index('ticker', inplace=True)
        
        # Load user portfolios if provided (standard format)
        if user_portfolios_path and os.path.exists(user_portfolios_path):
            self.user_portfolios = pd.read_csv(user_portfolios_path)
        else:
            self.user_portfolios = pd.DataFrame(columns=['user_id', 'ticker', 'weight'])
            
        # Load unique portfolios if provided
        if unique_portfolios_path and os.path.exists(unique_portfolios_path):
            self.load_unique_portfolios(unique_portfolios_path)
    
    def load_unique_portfolios(self, file_path):
        """Load and parse the unique portfolios format."""
        df = pd.read_csv(file_path)
        
        # Convert string representations of lists to actual lists
        df['ticker_list'] = df['ticker'].apply(ast.literal_eval)
        df['weight_list'] = df['weight'].apply(ast.literal_eval)
        
        self.unique_portfolios = df
    
    def prepare_features(self):
        """
        Prepare and normalize stock features for similarity calculations.
        """
        # Select numeric features
        features_df = self.stocks_data[self.feature_columns].copy()
        
        # Handle missing values
        features_df = features_df.fillna(features_df.mean())
        
        # Normalize features
        self.stock_features = self.scaler.fit_transform(features_df)
        
        # Create similarity matrix
        self.similarity_matrix = cosine_similarity(self.stock_features)
    
    def expand_user_portfolio(self, user_id):
        """
        Expand a user's portfolio from the unique format to the standard format.
        
        Parameters:
        user_id (str): The ID of the user
        
        Returns:
        pd.DataFrame: The expanded user portfolio
        """
        if self.unique_portfolios is None:
            raise ValueError("No unique portfolios data loaded")
            
        user_data = self.unique_portfolios[self.unique_portfolios['user_id'] == user_id]
        
        if user_data.empty:
            raise ValueError(f"User {user_id} not found in unique portfolios data")
            
        user_row = user_data.iloc[0]
        tickers = user_row['ticker_list']
        weights = user_row['weight_list']
        
        expanded_df = pd.DataFrame({
            'user_id': [user_id] * len(tickers),
            'ticker': tickers,
            'weight': weights
        })
        
        return expanded_df
        
    def create_user_profile(self, user_input):
        """
        Create a user profile based on their portfolio weights.
        
        Parameters:
        user_input (str or pd.DataFrame): User ID or DataFrame containing user portfolio
        
        Returns:
        np.ndarray: Vector representing user's stock preferences
        """
        if isinstance(user_input, str):
            # If user_id is provided, first check in unique portfolios
            if self.unique_portfolios is not None:
                try:
                    user_portfolio = self.expand_user_portfolio(user_input)
                except ValueError:
                    # If not found in unique portfolios, check in standard portfolios
                    if self.user_portfolios is not None:
                        user_portfolio = self.user_portfolios[self.user_portfolios['user_id'] == user_input]
                    else:
                        raise ValueError(f"User {user_input} not found in any portfolio data")
            else:
                # If unique portfolios not available, use standard portfolios
                if self.user_portfolios is not None:
                    user_portfolio = self.user_portfolios[self.user_portfolios['user_id'] == user_input]
                else:
                    raise ValueError(f"User {user_input} not found in any portfolio data")
        else:
            # If user_input is already a DataFrame
            user_portfolio = user_input
        
        # Create weighted average of stock features for the user's portfolio
        user_vector = np.zeros(len(self.feature_columns))
        total_weight = 0
        
        for _, row in user_portfolio.iterrows():
            ticker = row['ticker']
            weight = row['weight']
            
            if ticker in self.stocks_data.index:
                stock_idx = list(self.stocks_data.index).index(ticker)
                user_vector += weight * self.stock_features[stock_idx]
                total_weight += weight
        
        if total_weight > 0:
            user_vector /= total_weight
        
        return user_vector
    
    def analyze_portfolio_risks(self, user_id):
        """
        Analyze portfolio risks and provide alerts for overconcentration or lack of diversity.
        
        Parameters:
        user_id (str): The ID of the user
        
        Returns:
        dict: Risk analysis results with alerts
        """
        try:
            # Get user portfolio
            if self.unique_portfolios is not None:
                user_portfolio = self.expand_user_portfolio(user_id)
            else:
                user_portfolio = self.user_portfolios[self.user_portfolios['user_id'] == user_id]
                
            if user_portfolio.empty:
                raise ValueError(f"No portfolio data found for user {user_id}")
            
            # Calculate sector weights
            sector_weights = {}
            total_weight = 0
            
            for _, row in user_portfolio.iterrows():
                ticker = row['ticker']
                weight = row['weight']
                total_weight += weight
                
                if ticker in self.stocks_data.index:
                    stock_info = self.stocks_data.loc[ticker]
                    # Convert Series to dict if needed
                    if isinstance(stock_info, pd.Series):
                        stock_info = stock_info.to_dict()
                    
                    # Make sure sector is a string
                    sector = stock_info['sector']
                    if isinstance(sector, pd.Series):
                        sector = str(sector.iloc[0]) if not sector.empty else "Unknown"
                        
                    # Convert sector to string if it might be a complex object
                    if not isinstance(sector, str):
                        sector = str(sector)
                    
                    if sector in sector_weights:
                        sector_weights[sector] += weight
                    else:
                        sector_weights[sector] = weight
            
            # Normalize sector weights
            for sector in sector_weights:
                sector_weights[sector] = sector_weights[sector] / total_weight if total_weight > 0 else 0
            
            # Sort sectors by weight
            sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
            
            # Initialize risk analysis
            risk_analysis = {
                'alerts': [],
                'sector_concentration': sorted_sectors,
                'sector_count': len(sector_weights),
                'most_concentrated_sector': sorted_sectors[0] if sorted_sectors else None,
            }
            
            # Check for sector overconcentration
            for sector, weight in sorted_sectors:
                if weight > self.sector_concentration_threshold:
                    alert = {
                        'type': 'sector_overconcentration',
                        'sector': sector,
                        'weight': weight,
                        'message': f"Your portfolio is heavily concentrated in the {sector} sector ({weight*100:.1f}%). Consider diversifying to reduce risk."
                    }
                    risk_analysis['alerts'].append(alert)
            
            # Check for lack of sector diversity
            if len(sector_weights) < self.sector_count_min:
                alert = {
                    'type': 'lack_of_diversity',
                    'sector_count': len(sector_weights),
                    'message': f"Your portfolio only contains {len(sector_weights)} sectors. Consider investing in at least {self.sector_count_min} different sectors to improve diversification."
                }
                risk_analysis['alerts'].append(alert)
            
            # Check for high-beta concentration
            high_beta_stocks = []
            for _, row in user_portfolio.iterrows():
                ticker = row['ticker']
                weight = row['weight']
                
                if ticker in self.stocks_data.index:
                    stock_info = self.stocks_data.loc[ticker]
                    # Convert Series to dict if needed
                    if isinstance(stock_info, pd.Series):
                        stock_info = stock_info.to_dict()
                    
                    beta = stock_info['beta']
                    # Convert beta to float if it's a Series
                    if isinstance(beta, pd.Series):
                        beta = float(beta.iloc[0]) if not beta.empty and not pd.isna(beta.iloc[0]) else 0.0
                    
                    if isinstance(beta, (int, float)) and beta > 1.5:  # Stocks with beta > 1.5 are considered highly volatile
                        high_beta_stocks.append((ticker, beta, weight))
            
            high_beta_weight = sum(weight for _, _, weight in high_beta_stocks)
            if high_beta_weight > 0 and total_weight > 0 and high_beta_weight / total_weight > 0.3:  # If more than 30% in high-beta stocks
                alert = {
                    'type': 'high_volatility',
                    'high_beta_weight': high_beta_weight / total_weight,
                    'message': f"Your portfolio has {high_beta_weight/total_weight*100:.1f}% allocated to high-volatility stocks. This may lead to larger swings in portfolio value."
                }
                risk_analysis['alerts'].append(alert)
            
            return risk_analysis
            
        except Exception as e:
            print(f"Debug - Error in analyze_portfolio_risks: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a simple structure instead of raising an error
            return {
                'alerts': [],
                'sector_concentration': [],
                'sector_count': 0,
                'error': str(e)
            }

    def generate_diversification_recommendations(self, user_id, n=5):
        """
        Generate stock recommendations specifically for diversification.
        
        Parameters:
        user_id (str): The ID of the user
        n (int): Number of recommendations to generate
        
        Returns:
        list: Recommended stocks for diversification with explanations
        """
        try:
            # Get user portfolio
            if self.unique_portfolios is not None:
                user_portfolio = self.expand_user_portfolio(user_id)
            else:
                user_portfolio = self.user_portfolios[self.user_portfolios['user_id'] == user_id]
                
            if user_portfolio.empty:
                raise ValueError(f"No portfolio data found for user {user_id}")
            
            # Identify sectors in the user's portfolio
            portfolio_sectors = set()
            for _, row in user_portfolio.iterrows():
                ticker = row['ticker']
                if ticker in self.stocks_data.index:
                    stock_info = self.stocks_data.loc[ticker]
                    # Convert Series to dict if needed
                    if isinstance(stock_info, pd.Series):
                        stock_info = stock_info.to_dict()
                    
                    # Make sure sector is a string
                    sector = stock_info['sector']
                    if isinstance(sector, pd.Series):
                        sector = str(sector.iloc[0]) if not sector.empty else "Unknown"
                    
                    portfolio_sectors.add(sector)
            
            # Create a vector of the user's preferences
            user_vector = self.create_user_profile(user_portfolio)
            
            # Calculate similarities for all stocks
            similarities = cosine_similarity([user_vector], self.stock_features)[0]
            
            # Get indices of stocks that are NOT in the user's portfolio sectors
            diversification_indices = []
            portfolio_tickers = set(user_portfolio['ticker'])
            
            for idx, ticker in enumerate(self.stocks_data.index):
                stock_info = self.stocks_data.loc[ticker]
                # Convert Series to dict if needed
                if isinstance(stock_info, pd.Series):
                    stock_info = stock_info.to_dict()
                
                # Make sure sector is a string
                sector = stock_info['sector']
                if isinstance(sector, pd.Series):
                    sector = str(sector.iloc[0]) if not sector.empty else "Unknown"
                
                if sector not in portfolio_sectors and ticker not in portfolio_tickers:
                    diversification_indices.append((idx, similarities[idx], sector))
            
            # Sort by similarity (we still want stocks that match user preferences)
            diversification_indices.sort(key=lambda x: x[1], reverse=True)
            
            # Get top N recommendations
            recommendations = []
            sectors_added = set()
            
            for idx, similarity, sector in diversification_indices:
                # Ensure sector diversity in the recommendations
                if sector not in sectors_added or len(sectors_added) >= 3:
                    ticker = self.stocks_data.index[idx]
                    stock_info = self.stocks_data.loc[ticker]
                    # Convert Series to dict if needed
                    if isinstance(stock_info, pd.Series):
                        stock_info = stock_info.to_dict()
                    
                    # Generate a simple explanation
                    explanation = self._generate_simple_explanation(ticker, similarity, sector, is_diversification=True)
                    
                    # Create recommendation dict with proper handling of Series objects
                    company_name = stock_info['company_name']
                    if isinstance(company_name, pd.Series):
                        company_name = str(company_name.iloc[0]) if not company_name.empty else "Unknown"
                    
                    # Make sure sector is a string
                    sector_value = stock_info['sector']
                    if isinstance(sector_value, pd.Series):
                        sector_value = str(sector_value.iloc[0]) if not sector_value.empty else "Unknown"
                    
                    price = stock_info['price'] 
                    if isinstance(price, pd.Series):
                        price = float(price.iloc[0]) if not price.empty else 0.0
                    
                    market_cap = stock_info['market_cap']
                    if isinstance(market_cap, pd.Series):
                        market_cap = float(market_cap.iloc[0]) if not market_cap.empty else 0.0
                    
                    esg_score = stock_info['esg_score']
                    if isinstance(esg_score, pd.Series):
                        esg_score = float(esg_score.iloc[0]) if not esg_score.empty else 0.0
                    
                    recommendations.append({
                        'ticker': ticker,
                        'company_name': company_name,
                        'similarity_score': similarity,
                        'sector': sector_value,
                        'price': price,
                        'market_cap': market_cap,
                        'esg_score': esg_score,
                        'explanation': explanation,
                        'recommendation_type': 'diversification'
                    })
                    
                    sectors_added.add(sector)
                
                if len(recommendations) >= n:
                    break
            
            return recommendations
            
        except Exception as e:
            raise ValueError(f"Error generating diversification recommendations for user {user_id}: {str(e)}")

    def _generate_simple_explanation(self, ticker, similarity_score, sector, is_diversification=False):
        """
        Generate a simple, jargon-free explanation for why a stock is recommended.
        
        Parameters:
        ticker (str): The ticker symbol
        similarity_score (float): The similarity score
        sector (str): The stock's sector
        is_diversification (bool): Whether this is a diversification recommendation
        
        Returns:
        str: A simple explanation
        """
        stock_info = self.stocks_data.loc[ticker]
        # Convert Series to dict if needed
        if isinstance(stock_info, pd.Series):
            stock_info = stock_info.to_dict()
        
        # Basic explanation template
        if is_diversification:
            explanation = f"This stock is in the {sector} sector, which is not currently in your portfolio. "
            explanation += "Adding it would help spread your investments across more industries. "
        else:
            explanation = f"This stock is similar to what you already like, with a match score of {similarity_score:.2f}. "
        
        # Add information about the company
        price = stock_info['price']
        # Handle Series objects
        if isinstance(price, pd.Series):
            price = float(price.iloc[0]) if not price.empty and not pd.isna(price.iloc[0]) else 0.0
        
        price_val = float(price) if isinstance(price, (int, float, str)) else 0
        explanation += f"It's priced at ${price_val:.2f}. "
        
        # Add information about company size
        market_cap = stock_info['market_cap']
        # Handle Series objects
        if isinstance(market_cap, pd.Series):
            market_cap = float(market_cap.iloc[0]) if not market_cap.empty and not pd.isna(market_cap.iloc[0]) else 0.0
            
        market_cap_val = float(market_cap) if isinstance(market_cap, (int, float, str)) else 0
        
        if market_cap_val > 10000:
            size = "very large"
        elif market_cap_val > 1000:
            size = "large"
        elif market_cap_val > 100:
            size = "medium-sized"
        else:
            size = "smaller"
        explanation += f"It's a {size} company. "
        
        # Add information about ESG score if available
        esg_score = stock_info['esg_score']
        # Handle Series objects
        if isinstance(esg_score, pd.Series):
            esg_score = float(esg_score.iloc[0]) if not esg_score.empty and not pd.isna(esg_score.iloc[0]) else 0.0
            
        if isinstance(esg_score, (int, float)):
            esg_val = float(esg_score)
            if esg_val > 80:
                explanation += "It has excellent environmental and social practices. "
            elif esg_val > 60:
                explanation += "It has good environmental and social practices. "
        
        # Add information about volatility if available
        beta = stock_info['beta']
        # Handle Series objects
        if isinstance(beta, pd.Series):
            beta = float(beta.iloc[0]) if not beta.empty and not pd.isna(beta.iloc[0]) else 0.0
            
        if isinstance(beta, (int, float)):
            beta_val = float(beta)
            if beta_val > 1.5:
                explanation += "The stock price tends to change more than the overall market. "
            elif beta_val < 0.8:
                explanation += "The stock price tends to be more stable than the overall market. "
        
        return explanation

    def generate_recommendations(self, user_input, n=5, exclude_portfolio=True, include_explanations=True):
        """
        Generate stock recommendations based on user input.
        
        Parameters:
        user_input (str or pd.DataFrame): User ID, ticker, or DataFrame containing user portfolio
        n (int): Number of recommendations to generate
        exclude_portfolio (bool): Whether to exclude stocks already in the user's portfolio
        include_explanations (bool): Whether to include simple explanations
        
        Returns:
        list: Top N recommended stocks with similarity scores and explanations
        """
        if isinstance(user_input, str):
            # Check if input is a ticker
            if user_input in self.stocks_data.index:
                stock_idx = list(self.stocks_data.index).index(user_input)
                similarities = self.similarity_matrix[stock_idx]
                portfolio_tickers = [user_input] if exclude_portfolio else []
            else:
                # Input is a user ID
                try:
                    if self.unique_portfolios is not None:
                        user_portfolio = self.expand_user_portfolio(user_input)
                    else:
                        user_portfolio = self.user_portfolios[self.user_portfolios['user_id'] == user_input]
                        
                    if user_portfolio.empty:
                        raise ValueError(f"No portfolio data found for user {user_input}")
                        
                    user_vector = self.create_user_profile(user_portfolio)
                    similarities = cosine_similarity([user_vector], self.stock_features)[0]
                    
                    portfolio_tickers = set(user_portfolio['ticker']) if exclude_portfolio else []
                except ValueError as e:
                    raise ValueError(f"Error processing user {user_input}: {str(e)}")
        else:
            # If user_input is a portfolio or user profile vector
            if isinstance(user_input, pd.DataFrame):
                user_vector = self.create_user_profile(user_input)
                portfolio_tickers = set(user_input['ticker']) if exclude_portfolio else []
            else:
                user_vector = user_input
                portfolio_tickers = []
            
            # Calculate similarities between user profile and all stocks
            similarities = cosine_similarity([user_vector], self.stock_features)[0]
        
        # Get indices of stocks, sorted by similarity
        similar_indices = similarities.argsort()[::-1]
        
        # Filter out stocks in the user's portfolio if requested
        if exclude_portfolio and portfolio_tickers:
            similar_indices = [idx for idx in similar_indices 
                             if self.stocks_data.index[idx] not in portfolio_tickers]
        
        # Get top N recommendations
        top_n_indices = similar_indices[:n]
        recommendations = []
        
        for idx in top_n_indices:
            ticker = self.stocks_data.index[idx]
            similarity = similarities[idx]
            stock_info = self.stocks_data.loc[ticker]
            # Convert Series to dict if needed
            if isinstance(stock_info, pd.Series):
                stock_info = stock_info.to_dict()
            
            # Handle potential Series objects for each field
            company_name = stock_info['company_name']
            if isinstance(company_name, pd.Series):
                company_name = str(company_name.iloc[0]) if not company_name.empty else "Unknown"
            
            sector = stock_info['sector']
            if isinstance(sector, pd.Series):
                sector = str(sector.iloc[0]) if not sector.empty else "Unknown"
            
            price = stock_info['price']
            if isinstance(price, pd.Series):
                price = float(price.iloc[0]) if not price.empty and not pd.isna(price.iloc[0]) else 0.0
            
            market_cap = stock_info['market_cap']
            if isinstance(market_cap, pd.Series):
                market_cap = float(market_cap.iloc[0]) if not market_cap.empty and not pd.isna(market_cap.iloc[0]) else 0.0
            
            esg_score = stock_info['esg_score']
            if isinstance(esg_score, pd.Series):
                esg_score = float(esg_score.iloc[0]) if not esg_score.empty and not pd.isna(esg_score.iloc[0]) else 0.0
            
            rec_dict = {
                'ticker': ticker,
                'company_name': company_name,
                'similarity_score': similarity,
                'sector': sector,
                'price': price,
                'market_cap': market_cap,
                'esg_score': esg_score,
                'recommendation_type': 'standard'
            }
            
            # Add simple explanation if requested
            if include_explanations:
                explanation = self._generate_simple_explanation(
                    ticker, similarity, sector
                )
                rec_dict['explanation'] = explanation
            
            recommendations.append(rec_dict)
        
        return recommendations
    
    def get_user_portfolio_summary(self, user_id):
        """
        Get a summary of a user's portfolio.
        
        Parameters:
        user_id (str): The ID of the user
        
        Returns:
        dict: Summary statistics about the user's portfolio
        """
        try:
            if self.unique_portfolios is not None:
                user_portfolio = self.expand_user_portfolio(user_id)
            else:
                user_portfolio = self.user_portfolios[self.user_portfolios['user_id'] == user_id]
                
            if user_portfolio.empty:
                raise ValueError(f"No portfolio data found for user {user_id}")
                
            # Calculate portfolio statistics
            total_value = 0
            sector_weights = {}
            stocks_info = []
            
            for _, row in user_portfolio.iterrows():
                ticker = row['ticker']
                weight = row['weight']
                
                if ticker in self.stocks_data.index:
                    stock_info = self.stocks_data.loc[ticker].to_dict()  # Convert Series to dict
                    price = stock_info['price']
                    sector = stock_info['sector']
                    
                    # Add to sector weights
                    if sector in sector_weights:
                        sector_weights[sector] += weight
                    else:
                        sector_weights[sector] = weight
                    
                    # Add stock info
                    stocks_info.append({
                        'ticker': ticker,
                        'company_name': stock_info['company_name'],
                        'weight': weight,
                        'price': price,
                        'sector': sector
                    })
                    
                    # Add to total value (assuming weight is percentage allocation)
                    total_value += weight * price
            
            # Sort sectors by weight
            sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate portfolio statistics
            summary = {
                'user_id': user_id,
                'total_value': total_value,
                'num_stocks': len(stocks_info),
                'sector_distribution': sorted_sectors,
                'stocks': sorted(stocks_info, key=lambda x: x['weight'], reverse=True)
            }
            
            return summary
        except Exception as e:
            raise ValueError(f"Error generating portfolio summary for user {user_id}: {str(e)}")
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Parameters:
        filepath (str): Path to save the model
        """
        model_data = {
            'stocks_data': self.stocks_data,
            'user_portfolios': self.user_portfolios,
            'unique_portfolios': self.unique_portfolios,
            'stock_features': self.stock_features,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'similarity_matrix': self.similarity_matrix
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Parameters:
        filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)
        self.stocks_data = model_data['stocks_data']
        self.user_portfolios = model_data.get('user_portfolios')
        self.unique_portfolios = model_data.get('unique_portfolios')
        self.stock_features = model_data['stock_features']
        self.feature_columns = model_data['feature_columns']
        self.scaler = model_data['scaler']
        self.similarity_matrix = model_data['similarity_matrix'] 