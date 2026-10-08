from flask import Flask, render_template, jsonify
import yfinance as yf
import plotly.graph_objs as go
import plotly.utils
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error

app = Flask(__name__)

# 공통 설정
TICKERS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD']
COLORS = ['#ff9900', '#00b3ff', '#ffd700', '#00ff88', '#ff00ff']

def get_crypto_chart():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)
    
    fig = go.Figure()
    
    for i, ticker in enumerate(TICKERS):
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
        
        if not data.empty and 'Close' in data.columns:
            close_data = data['Close'].dropna()
            if isinstance(close_data, pd.DataFrame):
                close_data = close_data.iloc[:, 0]
            
            x_vals = close_data.index.strftime('%Y-%m-%d').tolist()
            y_vals = close_data.tolist()
            
            fig.add_trace(go.Scatter(
                x=x_vals, 
                y=y_vals, 
                mode='lines', 
                name=ticker.replace('-USD', ''), 
                line=dict(color=COLORS[i], width=2)
            ))
    
    fig.update_layout(
        title='Top 5 Cryptocurrencies Price Over Last 10 Years',
        xaxis_title='Date',
        yaxis_title='Price (USD) - Log Scale',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_yaxes(type="log", autorange=True)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    graphJSON = get_crypto_chart()
    return render_template('index.html', graphJSON=graphJSON)

@app.route('/api/update')
def update_data():
    return get_crypto_chart()

@app.route('/api/predict')
def predict_future():
    end_date = datetime.now()
    # Train: 3 years ago ~ 1 year ago, Test: 1 year ago ~ yesterday
    # We need slightly more than 3 years to create shift features for the first few days
    start_date = end_date - timedelta(days=365*3 + 10) 
    
    results = {}
    
    for i, ticker in enumerate(TICKERS):
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
        if data.empty or 'Close' not in data.columns:
            continue
            
        close_data = data['Close'].dropna()
        if isinstance(close_data, pd.DataFrame):
            close_data = close_data.iloc[:, 0]
            
        df = pd.DataFrame({'Close': close_data})
        
        # Feature: D-1, D-2, D-3, D-4, D-5
        for j in range(1, 6):
            df[f'D-{j}'] = df['Close'].shift(j)
            
        df = df.dropna()
        
        one_year_ago = end_date - timedelta(days=365)
        # Training data: 3 years ago ~ 1 year ago
        train = df[df.index < one_year_ago]
        # Test data: 1 year ago ~ yesterday
        # To strictly be up to yesterday, we can just use df[df.index >= one_year_ago] (it naturally goes up to latest available)
        test = df[df.index >= one_year_ago]
        
        features = [f'D-{j}' for j in range(1, 6)]
        
        X_train = train[features]
        y_train = train['Close']
        
        X_test = test[features]
        y_test = test['Close']
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
            
        # 여러 알고리즘을 정의하여 코인별로 가장 성능(MAPE)이 좋은 "별도 모델"을 탐색 및 선택합니다.
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from xgboost import XGBRegressor
        
        models_to_try = {
            'DecisionTree': DecisionTreeRegressor(random_state=42),
            'RandomForest': RandomForestRegressor(random_state=42, n_estimators=50),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42, n_estimators=50),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge()
        }
        
        best_model_name = None
        best_model = None
        best_mape = float('inf')
        
        for m_name, m in models_to_try.items():
            m.fit(X_train, y_train)
            preds = m.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, preds) * 100
            
            if mape < best_mape:
                best_mape = mape
                best_model = m
                best_model_name = m_name
        
        # Predict Tomorrow using the BEST model
        last_5 = df['Close'].iloc[-5:].values
        next_features = np.array([last_5[-1], last_5[-2], last_5[-3], last_5[-4], last_5[-5]]).reshape(1, -1)
        
        predicted_value = best_model.predict(next_features)[0]
        
        coin_name = ticker.replace('-USD', '')
        results[coin_name] = {
            'predicted_price': predicted_value,
            'mape': best_mape,
            'color': COLORS[i],
            'best_model': best_model_name
        }
        
    return jsonify(results)

@app.route('/api/kosis')
def fetch_kosis():
    url = "https://kosis.kr/openapi/Param/statisticsParameterData.do?method=getList&apiKey=N2QxMTRkNWRlNmM4N2YyMTc1MzM5MmE4MTU5ZTZkOTU=&itmId=13103130641T1+&objL1=ALL&objL2=ALL&objL3=&objL4=&objL5=&objL6=&objL7=&objL8=&format=json&jsonVD=Y&prdSe=M&newEstPrdCnt=3&orgId=343&tblId=DT_343_2010_S0140"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data)
        file_name = "kosis.xlsx"
        df.to_excel(file_name, index=False)
        
        # 유의미한 데이터 추출: ETF 거래대금(Trading Value)의 월별 추이
        df['DT'] = pd.to_numeric(df['DT'], errors='coerce')
        value_df = df[df['C1_NM_ENG'] == 'Trading Value'].copy()
        
        # 월별(PRD_DE) 총 거래대금 합산
        monthly_vol = value_df.groupby('PRD_DE')['DT'].sum().reset_index()
        monthly_vol = monthly_vol.sort_values('PRD_DE')
        
        # 날짜 포맷(YYYYMM -> YYYY-MM) 적용
        x_vals = monthly_vol['PRD_DE'].apply(lambda x: f"{str(x)[:4]}-{str(x)[4:6]}").tolist()
        y_vals = monthly_vol['DT'].tolist()
        
        return jsonify({
            "status": "success",
            "file_name": file_name,
            "rows": len(df),
            "columns": len(df.columns),
            "chart_data": {
                "x": x_vals,
                "y": y_vals
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
