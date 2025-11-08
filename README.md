# Stock analysis 📈

Streamlit application leveraging large language models (LLMs) to automate financial analysis. Performed RAG over a financial dataset, using LLMs to generate financial reports, scrape websites, and extract sentiment from news articles. By regenerating prompts, dynamically route models, and generate structured outputs.

### Asynchronous/Parallel Programming architecture to load 9998 stock tickers with their relevant metadata from yfinance to Pinecone Index

![image](https://github.com/user-attachments/assets/a94b044c-cf98-4b64-90a6-270da7ee6849)

## Features

- **Stock Search**: Search for stocks by description, sector, or characteristics.
- **Stock Cards**: Display detailed information about each stock, including market cap, price, growth, and analyst recommendations.
- **Sector Distribution**: Visualize the distribution of stocks across different sectors.
- **Market Cap Comparison**: Compare the market capitalization of selected stocks.
- **Price Performance**: Analyze the 1-year price performance of stocks.
- **AI Analysis**: Get AI-generated insights and analysis on selected stocks.
- **Geographic Distribution**: View the geographic distribution of companies.
- **Beta and Dividend Yield Comparison**: Compare beta values and dividend yields of stocks.

## Local Setup

### Prerequisites

- Python 3.9 or higher
- A Groq API key (get one at https://console.groq.com/keys)
- A Pinecone API key (get one at https://app.pinecone.io/)

### Running the App Locally

1. **Activate the virtual environment:**

   ```bash
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate  # Windows
   ```

2. **Install/update dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   **Note:** If you encounter a dependency conflict with `pinecone-plugin-inference`, the requirements.txt has been updated to fix this. Make sure to reinstall:

   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Configure API keys:**

   Create or edit `.streamlit/secrets.toml`:

   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   PINECONE_API_KEY = "your_pinecone_api_key_here"
   ```

4. **Run the Streamlit app:**

   ```bash
   streamlit run streamlit_app.py
   ```

   The app will open in your browser at `http://localhost:8501`

### Troubleshooting

- **Dependency conflicts:** If you see errors about `pinecone-plugin-inference`, run `pip install --upgrade -r requirements.txt` to ensure compatible versions are installed.
- **API key errors:** Make sure your `.streamlit/secrets.toml` file is in the correct location and contains valid API keys.
- **Port already in use:** Use `streamlit run streamlit_app.py --server.port 8502` to use a different port.

## ToDos

- **Research Automation**: Build a system that can find relevant stocks based on natural language queries from the user (e.g. "What are companies that build data centers?"). All stocks on the New York Stock Exchange must also be searchable by metrics such as Market Capitalization, Volume, Sector, and more.
- **Market Firehose**: Build a system that can handle 100 articles per minute. Your system should be able to process unstructured text articles and parse out the publisher, author, date, title, body, related sector. This should include an API and database schema. It must be a highly extensible system that can support articles from many different feeds, allows others to subscribe to the system to receive structured articles, and must operate as close to real time as possible while being able to handle processing hundreds of articles per minute.
- **Market Analysis**: Build a system that can process articles received from the market firehose.It should determine which companies are related to the article, a sentiment per company (positive or negative), and an event type classification for each article. Remember, this system must operate as close to real time as possible, and your system may receive hundreds of articles per minute.
- Free Worldwide News API
- Yahoo Finance API in Python
- US Stocks that will benefit with the incoming Trump Administration
- Data centers - META recently invested in datacenter for their llama open source model
- Jim Simons - successful quant investor billionaire with mathematical models and algorithms to make investment gains from market inefficiencies
