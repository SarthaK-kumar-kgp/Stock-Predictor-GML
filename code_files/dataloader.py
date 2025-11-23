class DataLoader:
    def __init__(self, period, interval, start_date=None, end_date=None,
                 sleep_between: float = 1.0, jitter: float = 0.5,
                 max_retries: int = 3, backoff_factor: float = 2.0, finnhub_api_key: str = None):
        """Initialize DataLoader with optional rate-limiting parameters:
        sleep_between: base seconds to sleep between requests
        jitter: max random jitter added/subtracted from sleep_between
        max_retries: number of retries on transient failures
        backoff_factor: multiplier for exponential backoff on retries
        """
        self.finnhub_api_key = finnhub_api_key
        self.period = period
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        # rate limiting / retry params
        self.sleep_between = sleep_between
        self.jitter = jitter
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        if finnhub_api_key is None:
            raise ValueError("Finnhub API key required for sentiment fetching.")
        self.client = finnhub.Client(api_key=self.finnhub_api_key)
        self.analyzer = SentimentIntensityAnalyzer()
    def _safe_call(self, func, *args, **kwargs):
        """Call API with retries, exponential backoff, and jitter."""
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries:
                    print(f"Failed after {attempt} attempts: {e}")
                    return None
                delay = (self.backoff_factor ** (attempt - 1)) * self.sleep_between + random.uniform(-self.jitter, self.jitter)
                print(f"API error ({e}), retrying in {max(0, delay):.2f}s...")
                time.sleep(max(0, delay))
        return None

    def expand_weekly_to_daily(self, weekly_df):
        """
        Expand weekly sentiment data to daily rows for merging.
        Each week_start row is repeated for all 7 days in that week.
        Assumes 'week_start' is a datetime.date or string, and other columns are to be copied.
        """
        daily_rows = []
        for _, row in weekly_df.iterrows():
            week_start = pd.to_datetime(row['week_start']).date()
            for i in range(7):
                day = week_start + timedelta(days=i)
                daily_row = row.copy()
                daily_row['Date'] = day
                daily_rows.append(daily_row)
        daily_df = pd.DataFrame(daily_rows)
        daily_df.drop(columns=['week_start'], inplace=True)
        return daily_df

    def _get_ticker_(self):
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers).content
        sp500_table = pd.read_html(html)[0]
        symbols = " ".join(sp500_table["Symbol"].tolist())
        sp500 = yf.Tickers(symbols)

        result = []
        for t in sp500.tickers:
            try:
                info = sp500.tickers[t].info
                result.append({
                    "ticker": t,
                    "name": info.get("shortName"),
                    "marketCap": info.get("marketCap"),
                    "price": info.get("regularMarketPrice")
                })
            except Exception:
                pass
        return pd.DataFrame(result)

    def _get_stock_prices_(self, tickers: List[str]) -> pd.DataFrame:
        all_data = []
        for ticker in tqdm(tickers, desc="Downloading stock data"):
            attempt = 0
            while attempt <= self.max_retries:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period=self.period, interval=self.interval)

                    if hist.empty:
                        print(f"No data found for {ticker}")
                        break  # nothing to retry for empty data

                    hist.reset_index(inplace=True)
                    hist['Ticker'] = ticker  # tag each record with its ticker

                    # Optional: keep only relevant columns
                    # guard in case some tickers have different columns
                    cols = [c for c in ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'] if c in hist.columns]
                    hist = hist[cols]

                    all_data.append(hist)
                    break  # success -> exit retry loop

                except Exception as e:
                    attempt += 1
                    if attempt > self.max_retries:
                        print(f"Error fetching {ticker} after {self.max_retries} retries: {e}")
                        break
                    # exponential backoff with jitter before retrying
                    backoff = (self.backoff_factor ** (attempt - 1)) * self.sleep_between
                    jitter = random.uniform(-self.jitter, self.jitter)
                    sleep_time = max(0.0, backoff + jitter)
                    print(f"Transient error fetching {ticker} (attempt {attempt}/{self.max_retries}), retrying in {sleep_time:.2f}s: {e}")
                    time.sleep(sleep_time)

            # Respect a minimum pause between successive tickers to avoid rate limits
            pause = max(0.0, self.sleep_between + random.uniform(-self.jitter, self.jitter))
            time.sleep(pause)

        if not all_data:
            print("No data fetched. Check tickers or network.")
            return pd.DataFrame()

        stock_prices = pd.concat(all_data, ignore_index=True)
        return stock_prices

    def _get_industry_(self,ticker:pd.DataFrame):
        """
        Args:
            ticker_dataframe (pd.DataFrame): must contain a 'ticker' column
        Returns:
            pd.DataFrame: columns ['ticker', 'industry', 'sector']
        """
        if "ticker" not in ticker.columns:
            raise ValueError("Input DataFrame must contain a 'ticker' column")
        records = []
        for ticker in tqdm(ticker["ticker"], desc="Fetching industry data"):
            try:
                info = yf.Ticker(ticker).info
                industry = info.get("industry")
                sector = info.get("sector")
            except Exception as e:
                industry = None
                sector = None
                print(f"Failed to fetch info for {ticker}: {e}")

            records.append({
                "ticker": ticker,
                "industry": industry,
                "sector": sector
            })

        return pd.DataFrame(records)

    def _get_volatility_(self):
        vix = yf.Ticker("^VIX")
        vix_data = vix.history(period=self.period,interval = self.interval)  # daily OHLC + volume
        # vix.reset_index()
        vix_data.reset_index(inplace=True)
        vix_data['Date'] = pd.to_datetime(vix_data['Date'])
        vix_data = vix_data[['Date', 'Open', 'High', 'Low', 'Close']]
        vix_data.rename(columns={
            'Open': 'VIX_Open',
            'High': 'VIX_High',
            'Low': 'VIX_Low',
            'Close': 'VIX_Close'
        }, inplace=True)
        return vix_data

    def _get_sentiment_(self, tickers_df: pd.DataFrame, start_date=None, end_date=None):
        """
        Fetch weekly sentiment + headlines for a list of tickers between start_date and end_date
        Args:
            tickers_df: DataFrame with columns ['ticker', 'stock_name'],start_date: 'YYYY-MM-DD' or datetime.date
            end_date: 'YYYY-MM-DD' or datetime.date
        Returns:DataFrame with columns [ticker, week_start, avg_sentiment, headline_count, headlines_used]
        """
        # Handle date parsing
        if end_date is None:
            end_date = datetime.today().date()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        if start_date is None:
            start_date = end_date - timedelta(days=30)
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()

        results = []

        for _, row in tickers_df.iterrows():
            tkr, name = row['ticker'], row['name']
            print(f"\nFetching news for {tkr} ({name}) from {start_date} → {end_date}")

            news = self._safe_call(
                self.client.company_news,
                tkr,
                _from=start_date.isoformat(),
                to=end_date.isoformat()
            )
            if not news:
                continue
            if name!=None:
                search_terms = [tkr.lower()] + [w.lower() for w in name.split() if len(w) > 2]
            else:
                search_terms = tkr.lower()
            relevant = [n for n in news if any(term in n['headline'].lower() for term in search_terms)]
            if not relevant:
                continue

            temp = pd.DataFrame([{
                'ticker': tkr,
                'date': pd.to_datetime(n['datetime'], unit='s').date(),
                'headline': n['headline']
            } for n in relevant])

            temp['sentiment'] = temp['headline'].apply(lambda h: self.analyzer.polarity_scores(h)['compound'])
            temp['week_start'] = temp['date'] - pd.to_timedelta(temp['date'].apply(lambda d: d.weekday()), unit='D')

            weekly = temp.groupby('week_start').agg(
                avg_sentiment=('sentiment', 'mean'),
                headline_count=('headline', 'count'),
                headlines_used=('headline', list)
            ).reset_index()
            weekly['ticker'] = tkr

            results.append(weekly)

            # Pause between tickers
            time.sleep(max(0, self.sleep_between + random.uniform(-self.jitter, self.jitter)))

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def _get_all_data(self):
        tickers = self._get_ticker_()

        stock_prices = self._get_stock_prices_(tickers['ticker'].tolist())
        industry_info = self._get_industry_(tickers)
        vix_data = self._get_volatility_()
        dataframe = stock_prices.merge(industry_info, left_on='Ticker', right_on='ticker', how='left')
        dataframe['Date'] = pd.to_datetime(dataframe['Date']).dt.date
        vix_data['Date'] = pd.to_datetime(vix_data['Date']).dt.date
        dataframe = dataframe.merge(vix_data, on='Date', how='left')
        start_date = str(dataframe.Date.min())
        end_date = str(dataframe.Date.max())
        sentiment_data = self._get_sentiment_(tickers, start_date, end_date)
        sentiment_daily = self.expand_weekly_to_daily(sentiment_data)
        dataframe = dataframe.merge(sentiment_daily, on=['Date', 'ticker'], how='left')
        return dataframe
