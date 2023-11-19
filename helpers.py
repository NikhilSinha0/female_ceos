import numpy as np
import pandas as pd
import yfinance as yf
import sys
from datetime import datetime, timedelta

ceos_df = pd.read_csv("CEOS_Kaggle.csv")
ceos_df.set_index("CEO", inplace=True)
companies_df = pd.read_csv("fortune1000_2023.csv")
companies_df = companies_df[companies_df['Ticker'].notna()]
companies_df.set_index("Ticker", inplace=True)

def build_industry_map():
    industry_map = {}
    for name, row in companies_df.iterrows():
        industry, ticker = row.loc["Industry"], name
        if industry not in industry_map:
            industry_map[industry] = []
        industry_map[industry].append(ticker)
    return industry_map

industry_map = build_industry_map()

def get_prices_by_ceo_name(name):
    try:
        ceo_data = ceos_df.loc[name,["Ticker", "hire date (female only)"]].to_numpy().tolist()
        industry = companies_df.loc[ceo_data[0],"Industry"]
        industry_list = industry_map[industry]
        hire_date = datetime.strptime(ceo_data[1], '%m/%d/%Y').date()
        before_hire_date = hire_date - timedelta(days=365)
        # Lazy rn, add a check that the hire date is more than a year in the past
        after_hire_date = hire_date + timedelta(days=365)
        if after_hire_date > datetime.now().date():
            after_hire_date = datetime.now().date() - timedelta(days=1)
        industry_data_before = yf.download(
            industry_list,
            start=before_hire_date.strftime("%Y-%m-%d"),
            end=(hire_date - timedelta(days=1)).strftime("%Y-%m-%d")
        )['Adj Close']
        industry_data_before.dropna(axis=1, inplace=True)
        company_data_before = yf.download(
            [ceo_data[0]],
            start=before_hire_date.strftime("%Y-%m-%d"),
            end=(hire_date - timedelta(days=1)).strftime("%Y-%m-%d")
        )['Adj Close']
        industry_data_after = yf.download(
            industry_list,
            start=hire_date.strftime("%Y-%m-%d"),
            end=after_hire_date.strftime("%Y-%m-%d")
        )['Adj Close']
        industry_data_after.dropna(axis=1, inplace=True)
        company_data_after = yf.download(
            [ceo_data[0]],
            start=hire_date.strftime("%Y-%m-%d"),
            end=after_hire_date.strftime("%Y-%m-%d")
        )['Adj Close']
        return restore_index(collapse_cols_to_avg(convert_to_pct(industry_data_before))), restore_index(collapse_cols_to_avg(convert_to_pct(industry_data_after))), restore_index(convert_to_pct(company_data_before)), restore_index(convert_to_pct(company_data_after))
    except KeyError:
        print(f"No data found for {name}")
        return None, None, None, None
    
def convert_to_pct(frame):
    if len(frame.shape) == 1:
        return frame.apply(lambda x: (x/(frame.iloc[0]) - 1) * 100)
    f = frame.apply(lambda x: x.div(frame.iloc[0]).subtract(1).mul(100), axis=1)
    print(f.head())
    return f

def collapse_cols_to_avg(frame):
    frame['industry_avg'] = frame.mean(axis=1)
    frame = frame[['industry_avg']]
    return frame

def restore_index(frame):
    if len(frame.shape) == 1:
        frame = frame.to_frame()
    frame.insert(0, "Day", [i for i in range(frame.shape[0])], True)
    frame.set_index("Day", inplace=True)
    return frame

if __name__ == '__main__':
    # Test
    a, b, c, d = get_prices_by_ceo_name("Karen S. Lynch")
    if not isinstance(a, pd.DataFrame):
        sys.exit(1)
    print(a.head())
    print(b.head())
    print(c.head())
    print(d.head())