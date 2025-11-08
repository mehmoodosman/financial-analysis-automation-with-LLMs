from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
from groq import Groq
import time
import json
import requests
from requests.exceptions import HTTPError

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Stock Analysis", page_icon="📈")

def fetch_stock_data(ticker, max_retries=3):
    """
    Fetch stock data with retry logic to handle JSON decode errors, rate limiting (429), and other API issues.
    """
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            # Add a longer delay to avoid rate limiting (Yahoo Finance is strict)
            time.sleep(1.5)
            info = stock.info
            
            # Validate that we got some data
            if not info or len(info) == 0:
                raise ValueError("Empty info dictionary returned")
            
            return {
                "name": info.get("shortName", ticker),
                "summary": info.get("longBusinessSummary", "Data unavailable"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
                "revenue_growth": info.get("revenueGrowth", "N/A"),
                "recommendation": info.get("recommendationKey", "N/A"),
                "country": info.get("country", "N/A"),
                "beta": info.get("beta", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A")
            }
        except (json.JSONDecodeError, KeyError, ValueError, AttributeError) as e:
            if attempt < max_retries - 1:
                # Longer wait for rate limiting issues
                wait_time = (2 ** attempt) * 3  # Exponential backoff: 3, 6, 12 seconds
                time.sleep(wait_time)
                continue
            else:
                # Return default data structure on final failure
                st.warning(f"⚠️ Could not fetch complete data for {ticker}. Using default values.")
                return {
                    "name": ticker,
                    "summary": "Data unavailable - API error",
                    "sector": "N/A",
                    "industry": "N/A",
                    "market_cap": "N/A",
                    "price": "N/A",
                    "revenue_growth": "N/A",
                    "recommendation": "N/A",
                    "country": "N/A",
                    "beta": "N/A",
                    "dividend_yield": "N/A"
                }
        except HTTPError as e:
            # Handle 429 Too Many Requests specifically
            if e.response and e.response.status_code == 429:
                if attempt < max_retries - 1:
                    # Much longer wait for rate limiting (429 errors)
                    wait_time = (2 ** attempt) * 10  # Exponential backoff: 10, 20, 40 seconds
                    st.info(f"⏳ Rate limited for {ticker}. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    st.warning(f"⚠️ Rate limited for {ticker} after {max_retries} attempts. Using default values.")
                    return {
                        "name": ticker,
                        "summary": "Data unavailable - Rate limited",
                        "sector": "N/A",
                        "industry": "N/A",
                        "market_cap": "N/A",
                        "price": "N/A",
                        "revenue_growth": "N/A",
                        "recommendation": "N/A",
                        "country": "N/A",
                        "beta": "N/A",
                        "dividend_yield": "N/A"
                    }
            else:
                # Other HTTP errors
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 3
                    time.sleep(wait_time)
                    continue
                else:
                    st.warning(f"⚠️ HTTP error for {ticker}: {str(e)[:100]}")
                    return {
                        "name": ticker,
                        "summary": "Data unavailable - HTTP error",
                        "sector": "N/A",
                        "industry": "N/A",
                        "market_cap": "N/A",
                        "price": "N/A",
                        "revenue_growth": "N/A",
                        "recommendation": "N/A",
                        "country": "N/A",
                        "beta": "N/A",
                        "dividend_yield": "N/A"
                    }
        except Exception as e:
            error_str = str(e)
            # Check if it's a 429 error in the error message
            if "429" in error_str or "Too Many Requests" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 10  # Longer wait for rate limiting
                    st.info(f"⏳ Rate limited for {ticker}. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    st.warning(f"⚠️ Rate limited for {ticker} after {max_retries} attempts. Using default values.")
                    return {
                        "name": ticker,
                        "summary": "Data unavailable - Rate limited",
                        "sector": "N/A",
                        "industry": "N/A",
                        "market_cap": "N/A",
                        "price": "N/A",
                        "revenue_growth": "N/A",
                        "recommendation": "N/A",
                        "country": "N/A",
                        "beta": "N/A",
                        "dividend_yield": "N/A"
                    }
            else:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 3
                    time.sleep(wait_time)
                    continue
                else:
                    st.warning(f"⚠️ Error fetching data for {ticker}: {str(e)[:100]}")
                    return {
                        "name": ticker,
                        "summary": "Data unavailable - Error occurred",
                        "sector": "N/A",
                        "industry": "N/A",
                        "market_cap": "N/A",
                        "price": "N/A",
                        "revenue_growth": "N/A",
                        "recommendation": "N/A",
                        "country": "N/A",
                        "beta": "N/A",
                        "dividend_yield": "N/A"
                    }

def format_large_number(num):
    if num == "N/A":
        return "N/A"
    try:
        num = float(num)
        if num >= 1e12:
            return f"${num/1e12:.1f}T"
        elif num >= 1e9:
            return f"${num/1e9:.1f}B"
        elif num >= 1e6:
            return f"${num/1e6:.1f}M"
        else:
            return f"${num:,.0f}"
    except:
        return "N/A"

def format_percentage(value):
    if value == "N/A":
        return "N/A"
    try:
        return f"{float(value)*100:.1f}%"
    except:
        return "N/A"

def display_stock_card(data, ticker):
    with st.container():
        st.markdown("""
            
        """, unsafe_allow_html=True)

        st.markdown(f"""
            {data['name']} ({ticker})

            {data['sector']} | {data['industry']}

            {data['summary'][:150]}...
        """, unsafe_allow_html=True)

        metrics = [
            {"label": "Market Cap", "value": format_large_number(data['market_cap'])},
            {"label": "Price", "value": format_large_number(data['price'])},
            {"label": "Growth", "value": format_percentage(data['revenue_growth'])},
            {"label": "Rating", "value": data['recommendation'].upper()}
        ]

        cols = st.columns(4)
        for col, metric in zip(cols, metrics):
            with col:
                st.metric(
                    label=metric['label'],
                    value=metric['value'],
                    delta=None,
                )

        st.markdown("", unsafe_allow_html=True)

def get_huggingface_embeddings(text):
    return sentence_model.encode(text)

def display_sector_distribution(stock_data):
    sector_counts = pd.Series([data['sector'] for data in stock_data]).value_counts()
    fig = px.pie(sector_counts, values=sector_counts.values, names=sector_counts.index, title="Sector Distribution")
    st.plotly_chart(fig, use_container_width=True)

def display_market_cap_comparison(stock_data, ticker_list):
    market_caps = [data['market_cap'] for data in stock_data]
    fig = px.bar(x=ticker_list, y=market_caps, labels={'x': 'Ticker', 'y': 'Market Cap'}, title="Market Cap Comparison")
    st.plotly_chart(fig, use_container_width=True)

def get_ai_analysis(stock_data, ticker_list):
    # Prepare the context for the AI
    context = "Analyze these stocks:\n\n"
    for i, data in enumerate(stock_data):
        context += f"{i+1}. {data['name']} ({ticker_list[i]})\n"
        context += f"   Sector: {data['sector']}\n"
        context += f"   Industry: {data['industry']}\n"
        context += f"   Market Cap: {format_large_number(data['market_cap'])}\n"
        context += f"   Revenue Growth: {format_percentage(data['revenue_growth'])}\n"
        context += f"   Analyst Recommendation: {data['recommendation'].upper()}\n\n"

    system_prompt = """You are a highly skilled stock analysis expert. Based on the information provided, deliver a concise and insightful analysis that includes:

    1. Common trends or patterns across the listed stocks.
    2. Key differences and unique characteristics of each stock.
    3. Notable opportunities and associated risks.
    4. Tailored investment strategies for each stock.
    5. A brief but impactful technical and fundamental analysis of each stock.
    Ensure the analysis is sharp, focused, and prioritizes actionable insights."""

    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            model="llama-3.3-70b-versatile"
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating AI analysis: {str(e)}"

def display_country_distribution(stock_data):
    country_counts = pd.Series([data['country'] for data in stock_data]).value_counts()
    fig = px.pie(country_counts, 
                 values=country_counts.values, 
                 names=country_counts.index, 
                 title="Geographic Distribution",
                 hole=0.4,  # Makes it a donut chart
                 color_discrete_sequence=px.colors.qualitative.Set3)  # Using Set3 color palette
    st.plotly_chart(fig, use_container_width=True)

def display_beta_comparison(stock_data, ticker_list):
    betas = [data['beta'] for data in stock_data]
    fig = px.bar(x=ticker_list, 
                 y=betas,
                 labels={'x': 'Ticker', 'y': 'Beta'},
                 title="Beta Comparison",
                 color_discrete_sequence=['rgb(102, 197, 204)'])  # Teal-like color
    fig.add_hline(y=1, 
                  line_dash="dash", 
                  line_color="red", 
                  annotation_text="Market Beta")
    st.plotly_chart(fig, use_container_width=True)

def display_dividend_yield_comparison(stock_data, ticker_list):
    dividend_yields = [data['dividend_yield'] if data['dividend_yield'] != "N/A" 
                      else 0 for data in stock_data]
    dividend_yields = [float(dy) * 100 if dy is not None else 0 for dy in dividend_yields]
    
    fig = px.bar(x=ticker_list, 
                 y=dividend_yields,
                 labels={'x': 'Ticker', 'y': 'Dividend Yield (%)'},
                 title="Dividend Yield Comparison",
                 color_discrete_sequence=['rgb(255, 166, 0)'])  # Orange color
    st.plotly_chart(fig, use_container_width=True)

# Initialize clients once at the top level
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
pinecone_index = pc.Index("stocks")
sentence_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

st.title("Ticker Analysis")

user_query = st.text_input("Search for tickers by description, sector, or characteristics:")

if st.button("🚀 Find Tickers", type="primary"):
    with st.spinner("Analyzing tickers..."):
        # Directly use the user query without enhancement
        query_embedding = get_huggingface_embeddings(user_query)
        search_results = pinecone_index.query(
            vector=query_embedding.tolist(),
            top_k=5,
            include_metadata=True,
            namespace="stock-descriptions"
        )

        ticker_list = [item['id'] for item in search_results['matches']]

        stock_data = []
        successful_tickers = []
        for i, ticker in enumerate(ticker_list):
            # Add delay between tickers to avoid rate limiting
            if i > 0:
                time.sleep(2.0)  # Wait 2 seconds between each ticker request
            data = fetch_stock_data(ticker)
            if data:
                stock_data.append(data)
                successful_tickers.append(ticker)

        # Check if we have any successful data fetches
        if len(stock_data) == 0:
            st.error("❌ Could not fetch data for any of the selected tickers. Please try again later.")
            st.stop()

        # Display stock cards
        for i in range(0, len(stock_data), 2):
            col1, col2 = st.columns(2)

            with col1:
                display_stock_card(stock_data[i], successful_tickers[i])

            if i + 1 < len(stock_data):
                with col2:
                    display_stock_card(stock_data[i+1], successful_tickers[i+1])

        # Display sector distribution
        if len(stock_data) > 0:
            display_sector_distribution(stock_data)

        # Display market cap comparison
        if len(stock_data) > 0:
            display_market_cap_comparison(stock_data, successful_tickers)

        # Create comparison chart
        if len(stock_data) > 0:
            st.subheader("Stock Price Comparison")
            fig = go.Figure()

            chart_tickers = []
            for ticker in successful_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    time.sleep(2.0)  # Longer delay to avoid rate limiting
                    hist_data = stock.history(period="1y")
                    
                    # Check if we got valid data
                    if hist_data.empty or len(hist_data) == 0:
                        st.warning(f"⚠️ No historical data available for {ticker}")
                        continue
                    
                    # Normalize the prices to percentage change
                    hist_data['Normalized'] = (hist_data['Close'] / hist_data['Close'].iloc[0] - 1) * 100

                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['Normalized'],
                        name=f"{ticker}",
                        mode='lines'
                    ))
                    chart_tickers.append(ticker)
                except HTTPError as e:
                    if e.response and e.response.status_code == 429:
                        st.warning(f"⚠️ Rate limited when fetching historical data for {ticker}. Skipping...")
                    else:
                        st.warning(f"⚠️ HTTP error fetching historical data for {ticker}: {str(e)[:100]}")
                    continue
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "Too Many Requests" in error_str:
                        st.warning(f"⚠️ Rate limited when fetching historical data for {ticker}. Skipping...")
                    else:
                        st.warning(f"⚠️ Could not fetch historical data for {ticker}: {str(e)[:100]}")
                    continue

            if chart_tickers:
                fig.update_layout(
                    title="1-Year Price Performance Comparison (%)",
                    yaxis_title="Price Change (%)",
                    template="plotly_dark",
                    height=500,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical data available for any of the selected tickers.")

        # Add AI Analysis section
        st.subheader("🤖 AI Analysis")
        with st.spinner("Generating AI analysis..."):
            # Prepare the context for the AI
            context = "Analyze these stocks:\n\n"
            for i, data in enumerate(stock_data):
                context += f"{i+1}. {data['name']} ({successful_tickers[i]})\n"
                context += f"   Sector: {data['sector']}\n"
                context += f"   Industry: {data['industry']}\n"
                context += f"   Market Cap: {format_large_number(data['market_cap'])}\n"
                context += f"   Revenue Growth: {format_percentage(data['revenue_growth'])}\n"
                context += f"   Analyst Recommendation: {data['recommendation'].upper()}\n\n"

            prompt = f"""You are a professional stock analyst. Based on the following information about these stocks:

{context}

Provide a concise analysis that includes:
1. Common themes or patterns among these stocks
2. Key differences or contrasts
3. Potential opportunities and risks
4. Investment strategy for each stock
5. Technical and Fundamental analysis for each stock

Keep the analysis very short and focused on the most important insights."""

            try:
                completion = groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model="llama-3.3-70b-versatile"
                )
                
                analysis = completion.choices[0].message.content
                st.markdown(analysis)
            except Exception as e:
                st.error(f"Error generating AI analysis: {str(e)}")

        # After the existing visualizations (around line 190), add:
        if len(stock_data) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                display_country_distribution(stock_data)
                display_beta_comparison(stock_data, successful_tickers)
            
            with col2:
                display_dividend_yield_comparison(stock_data, successful_tickers)