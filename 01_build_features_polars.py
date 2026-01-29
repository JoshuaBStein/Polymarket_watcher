import polars as pl

print('Importing Data now')

# 1. Use scan_csv for Lazy Evaluation (Does not load data yet)
trades = pl.scan_csv("/scratch/jbs263/Fintechstuff2/processed/trades.csv")
markets = pl.scan_csv("/scratch/jbs263/Fintechstuff2/markets.csv")

# 2. Preprocess & Join (Still lazy)
trades = trades.with_columns(
    pl.col("timestamp").str.to_datetime(),
    pl.col("usd_amount").cast(pl.Float64)
)

# Join efficiently: Polars will optimize this plan
data = trades.join(markets, left_on="market_id", right_on="id", how="left")

# 3. Define Regex Patterns
politics_pattern = r"(?i)election|president|senate|congress|democrat|republican|trump|biden|harris|nominee|poll|vote|cabinet|war|geopolitics|ukraine|israel|policy"
sports_pattern = r"(?i)nfl|nba|mlb|nhl|soccer|ufc|f1|formula 1|premier league|champions league|super bowl|finals|playoff|vs|tournament|cup|medal|tennis|golf"
crypto_pattern = r"(?i)bitcoin|btc|ethereum|eth|solana|sol|crypto|nft|airdrop|defi|token|stablecoin|usdc|usdt|tether|binance|coinbase|etf|memecoin|halving"

# 4. Calculate Global Reference Date
# We use a separate collect() for this scalar value, which is very fast and cheap
max_date = trades.select(pl.col("timestamp").max()).collect().item()

# 5. Feature Engineering Grouped by Maker
wallet_profile_query = data.group_by("maker").agg([
    
    # Tenure
    ((max_date - pl.col("timestamp").min()).dt.total_days()).alias("tenure_days"),
    
    # Aggression
    pl.col("usd_amount").mean().alias("aggression_avg_usd"),
    
    # Frequency
    (pl.len() / (
        (max_date - pl.col("timestamp").min()).dt.total_days().clip(lower_bound=1)
    )).alias("frequency_trades_daily"),
    
    # Domain Focus
    ((pl.col("market_slug").str.contains(sports_pattern) | pl.col("question").str.contains(sports_pattern)).sum() / pl.len() * 100).alias("domain_focus_sports_pct"),
    ((pl.col("market_slug").str.contains(politics_pattern) | pl.col("question").str.contains(politics_pattern)).sum() / pl.len() * 100).alias("domain_focus_politics_pct"),
    ((pl.col("market_slug").str.contains(crypto_pattern) | pl.col("question").str.contains(crypto_pattern)).sum() / pl.len() * 100).alias("domain_focus_crypto_pct"),
    
    # Diversification
    pl.col("market_id").n_unique().alias("diversification_markets_count")
])

# 6. Execute with Streaming
# streaming=True tells Polars to process in chunks, keeping RAM usage low.
print("Starting processing... this may take a few minutes but will not crash your RAM.")
wallet_profile_query.collect(streaming=True).write_csv("/scratch/jbs263/Fintechstuff2/wallet_profiles.csv")
print("Done! File saved to wallet_profiles.csv")
