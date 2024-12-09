from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
from groq import Groq

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Stock Analysis", page_icon="📈")

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "name": info.get("shortName", "N/A"),
        "summary": info.get("longBusinessSummary", "N/A"),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "price": info.get("currentPrice", "N/A"),
        "revenue_growth": info.get("revenueGrowth", "N/A"),
        "recommendation": info.get("recommendationKey", "N/A"),
        "country": info.get("country", "N/A"),
        "beta": info.get("beta", "N/A"),
        "dividend_yield": info.get("dividendYield", "N/A")
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

st.title("Stock Analysis")

user_query = st.text_input("Search for stocks by description, sector, or characteristics:")

if st.button("🚀 Find Stocks", type="primary"):
    with st.spinner("Analyzing stocks..."):
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
        for ticker in ticker_list:
            data = fetch_stock_data(ticker)
            if data:
                stock_data.append(data)

        for i in range(0, len(stock_data), 2):
            col1, col2 = st.columns(2)

            with col1:
                display_stock_card(stock_data[i], ticker_list[i])

            if i + 1 < len(stock_data):
                with col2:
                    display_stock_card(stock_data[i+1], ticker_list[i+1])

        # Display sector distribution
        if len(stock_data) > 0:
            display_sector_distribution(stock_data)

        # Display market cap comparison
        if len(stock_data) > 0:
            display_market_cap_comparison(stock_data, ticker_list)

        # Create comparison chart
        if len(stock_data) > 0:
            st.subheader("Stock Price Comparison")
            fig = go.Figure()

            for i, ticker in enumerate(ticker_list):
                stock = yf.Ticker(ticker)
                hist_data = stock.history(period="1y")

                # Normalize the prices to percentage change
                hist_data['Normalized'] = (hist_data['Close'] / hist_data['Close'].iloc[0] - 1) * 100

                fig.add_trace(go.Scatter(
                    x=hist_data.index,
                    y=hist_data['Normalized'],
                    name=f"{ticker}",
                    mode='lines'
                ))

            fig.update_layout(
                title="1-Year Price Performance Comparison (%)",
                yaxis_title="Price Change (%)",
                template="plotly_dark",
                height=500,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

        # Add AI Analysis section
        st.subheader("🤖 AI Analysis")
        with st.spinner("Generating AI analysis..."):
            # Prepare the context for the AI
            context = "Analyze these stocks:\n\n"
            for i, data in enumerate(stock_data):
                context += f"{i+1}. {data['name']} ({ticker_list[i]})\n"
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
                display_beta_comparison(stock_data, ticker_list)
            
            with col2:
                display_dividend_yield_comparison(stock_data, ticker_list)
