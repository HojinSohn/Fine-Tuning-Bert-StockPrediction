## Results & Insights
- Fine-tuned BERT for sequence classification on Reddit news headlines to predict DJIA stock price movements.  
- Used `yfinance` to fetch DJIA stock prices and labeled the dataset based on stock price changes (up or down).  
- Merged Reddit news headlines with corresponding stock price movements to create a training dataset (From 2008 to 2026).  
- Tried BERT finance tokens and standard BERT tokenization, finding that standard BERT tokenization performed better with 54% accuracy, just above a random guess.  
