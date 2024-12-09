# Stock Analysis Dashboard

📈 A Streamlit application for analyzing stock data, providing insights into market trends, sector distributions, and more.

## Features

- **Stock Search**: Search for stocks by description, sector, or characteristics.
- **Stock Cards**: Display detailed information about each stock, including market cap, price, growth, and analyst recommendations.
- **Sector Distribution**: Visualize the distribution of stocks across different sectors.
- **Market Cap Comparison**: Compare the market capitalization of selected stocks.
- **Price Performance**: Analyze the 1-year price performance of stocks.
- **AI Analysis**: Get AI-generated insights and analysis on selected stocks.
- **Geographic Distribution**: View the geographic distribution of companies.
- **Beta and Dividend Yield Comparison**: Compare beta values and dividend yields of stocks.

## ToDos

- **Research Automation**: Build a system that can find relevant stocks based on natural language queries from the user (e.g. "What are companies that build data centers?"). All stocks on the New York Stock Exchange must also be searchable by metrics such as Market Capitalization, Volume, Sector, and more.
- **Market Firehose**: Build a system that can handle 100 articles per minute. Your system should be able to process unstructured text articles and parse out the publisher, author, date, title, body, related sector. This should include an API and database schema. It must be a highly extensible system that can support articles from many different feeds, allows others to subscribe to the system to receive structured articles, and must operate as close to real time as possible while being able to handle processing hundreds of articles per minute.
- **Market Analysis**: Build a system that can process articles received from the market firehose.It should determine which companies are related to the article, a sentiment per company (positive or negative), and an event type classification for each article. Remember, this system must operate as close to real time as possible, and your system may receive hundreds of articles per minute.
- Free Worldwide News API
- Yahoo Finance API in Python
- US Stocks that will benefit with the incoming Trump Administration
- Data centers - META recently invested in datacenter for their llama open source model
- Jim Simons - successful quant investor billionaire with mathematical models and algorithms to make investment gains from market inefficiencies
