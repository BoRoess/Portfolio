
README - Chainlink take home exercise 

Boris Rebo

Metrics that measure underlying asset quality?

What is asset quality? 
-	It is my assumption the products are cryptocurrency pairs data feeds. Assets like BTC/BNB, BTC/USD, BNB/USD and many others 

What is Quality and how do we measure it? 
-	The quality of a currency are things like liquidity, spread, market depth at various price points, lending rates, price volatility, availability of assets on different infrastructures.  

Assignment Goal 

For this experiment I will test the BNB/BTC pair and focus specifically on the exchange rate between them. I will attempt to build a predictive model based on the time serious analysis of their relationship.

Assumptions/ Limitations

-High volatility 
-No seasonality 
-Moderate correlation of two currencies
-Due to limited data availability the test will be run from September 2023 to Match 2024
-Extraneous market circumstances such macroeconomic trends, political uncertainty etc. are not considered. 

Gathering artifacts 

-BNB/BTC price taken as a CSV from 01.09.2024 to 29.02.2024. from
 
https://www.investing.com/crypto/bnb/bnb-btc-historical-data

Tools

-Excel

Anaconda – Jupyter using following extensions 
-	NumPy
-	Pandas
-	OS
-	Matplotlib
-	Seaborn
-	Statsmodels
-	Pmdarima
-	Math
-	Scikit-learn
