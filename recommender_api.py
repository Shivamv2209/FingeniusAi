from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from improved_recommender import ImprovedStockRecommender
import os
from typing import List, Optional

import pandas as pd

app = FastAPI(title="FinGenius Recommender API")

# ----------------------------- CORS SETUP -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- LOAD MODEL -----------------------------
print("🔁 Loading recommender...")
recommender = ImprovedStockRecommender()

DATA_DIR = "stock_recommender_data"
STOCKS_CSV = os.path.join(DATA_DIR, "stocks_data.csv")
PORTFOLIO_CSV = os.path.join(DATA_DIR, "users_unique_portfolio.csv")

if not os.path.exists(STOCKS_CSV) or not os.path.exists(PORTFOLIO_CSV):
    raise FileNotFoundError("❌ Required data files not found. Please run data.py first.")

recommender.load_data(
    stocks_data_path=STOCKS_CSV,
    unique_portfolios_path=PORTFOLIO_CSV
)
recommender.prepare_features()

print("✅ Recommender is ready.")

# ----------------------------- MODELS -----------------------------
class TickerWeight(BaseModel):
    ticker: str
    weight: float

class RecommendationRequest(BaseModel):
    user_id: Optional[str] = None
    portfolio: Optional[List[TickerWeight]] = None

class CompareRequest(BaseModel):
    ticker1: str
    ticker2: str

# ----------------------------- ROUTES -----------------------------

@app.get("/")
def root():
    return {"message": "Welcome to the FinGenius Recommender API 🎯"}

@app.post("/recommend")
def recommend_stocks(req: RecommendationRequest):
    try:
        if req.portfolio:
            portfolio_df = pd.DataFrame([p.dict() for p in req.portfolio])
            recommendations = recommender.generate_recommendations(user_input=portfolio_df)
        elif req.user_id:
            recommendations = recommender.generate_recommendations(user_input=req.user_id)
        else:
            raise HTTPException(status_code=400, detail="Provide either user_id or portfolio.")
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/compare")
def compare_stocks(req: CompareRequest):
    try:
        t1 = req.ticker1.upper()
        t2 = req.ticker2.upper()

        if t1 not in recommender.stocks_data.index or t2 not in recommender.stocks_data.index:
            raise HTTPException(status_code=404, detail="One or both tickers not found.")

        stock1 = recommender.stocks_data.loc[t1].to_dict()
        stock2 = recommender.stocks_data.loc[t2].to_dict()

        # Get similarity score
        index1 = recommender.stocks_data.index.get_loc(t1)
        index2 = recommender.stocks_data.index.get_loc(t2)
        similarity = float(recommender.similarity_matrix[index1][index2])

         # ========== VERDICT LOGIC ==========
        verdict = []

        verdict = []

        if stock1.get("sharpe_ratio", 0) > stock2.get("sharpe_ratio", 0):
            verdict.append(f"{t1} gives better performance for the amount of risk it takes.")
        elif stock1.get("sharpe_ratio", 0) < stock2.get("sharpe_ratio", 0):
            verdict.append(f"{t2} gives better performance for the amount of risk it takes.")

        if stock1.get("beta", 0) < stock2.get("beta", 0):
            verdict.append(f"{t1} tends to have a more stable price than {t2}.")
        elif stock1.get("beta", 0) > stock2.get("beta", 0):
            verdict.append(f"{t2} tends to have a more stable price than {t1}.")

        if stock1.get("esg_score", 0) > stock2.get("esg_score", 0):
            verdict.append(f"{t1} scores better on environmental and social responsibility.")
        elif stock1.get("esg_score", 0) < stock2.get("esg_score", 0):
            verdict.append(f"{t2} scores better on environmental and social responsibility.")

        if stock1.get("roe", 0) > stock2.get("roe", 0):
            verdict.append(f"{t1} uses investor money more efficiently than {t2}.")
        elif stock1.get("roe", 0) < stock2.get("roe", 0):
            verdict.append(f"{t2} uses investor money more efficiently than {t1}.")

        combined_verdict = " ".join(verdict) if verdict else "Both stocks perform similarly across key factors."


        return {
            "ticker1": t1,
            "company1": stock1.get("company_name", "N/A"),
            "ticker2": t2,
            "company2": stock2.get("company_name", "N/A"),
            "similarity_score": round(similarity, 3),
            "metrics": {
                "price": [stock1.get("price"), stock2.get("price")],
                "market_cap": [stock1.get("market_cap"), stock2.get("market_cap")],
                "pe_ratio": [stock1.get("pe_ratio"), stock2.get("pe_ratio")],
                "beta": [stock1.get("beta"), stock2.get("beta")],
                "roe": [stock1.get("roe"), stock2.get("roe")],
                "esg_score": [stock1.get("esg_score"), stock2.get("esg_score")],
                "dividend_yield": [stock1.get("dividend_yield"), stock2.get("dividend_yield")],
                "sharpe_ratio": [stock1.get("sharpe_ratio"), stock2.get("sharpe_ratio")],
            },
            "verdict": combined_verdict
        }


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
