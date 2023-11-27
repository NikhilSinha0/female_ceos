import numpy as np
import pandas as pd
import yfinance as yf
import sys
import math
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate

ceos_df = pd.read_csv("CEOS_Kaggle.csv")
ceos_df.set_index("CEO", inplace=True)
companies_df = pd.read_csv("fortune1000_2023.csv")
companies_df = companies_df[companies_df['Ticker'].notna()]
companies_df.set_index("Ticker", inplace=True)

def build_industry_map():
    industry_map = {}
    for name, row in companies_df.iterrows():
        industry, ticker = row.loc["Sector"], name
        if industry not in industry_map:
            industry_map[industry] = []
        industry_map[industry].append(ticker)
    return industry_map

industry_map = build_industry_map()

def get_prices_by_ceo_name(name):
    try:
        ceo_data = ceos_df.loc[name,["Ticker", "hire date (female only)"]].to_numpy().tolist()
        industry = companies_df.loc[ceo_data[0],"Sector"]
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
    
def get_continuous_prices_by_ceo_name(name):
    try:
        ceo_data = ceos_df.loc[name,["Ticker", "hire date (female only)"]].to_numpy().tolist()
        industry = companies_df.loc[ceo_data[0],"Sector"]
        industry_list = industry_map[industry]
        hire_date = datetime.strptime(ceo_data[1], '%m/%d/%Y').date()
        before_hire_date = hire_date - timedelta(days=365)
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
        if company_data_before.shape[0] == 0:
            print(f"Company was not listed before having a female CEO, skipping {name}")
            return None, None, None, None
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
        industry_base = industry_data_before.iloc[0]
        company_base = company_data_before.iloc[0]
        return restore_index(collapse_cols_to_avg(convert_to_pct_base(industry_data_before, industry_base))), restore_index(collapse_cols_to_avg(convert_to_pct_base(industry_data_after, industry_base))), restore_index(convert_to_pct_base(company_data_before, company_base)), restore_index(convert_to_pct_base(company_data_after, company_base))
    except KeyError:
        print(f"No data found for {name}")
        return None, None, None, None
    
def convert_to_pct(frame):
    if len(frame.shape) == 1:
        return frame.apply(lambda x: (x/(frame.iloc[0]) - 1) * 100)
    f = frame.apply(lambda x: x.div(frame.iloc[0]).subtract(1).mul(100), axis=1)
    return f

def convert_to_pct_base(frame, base):
    if len(frame.shape) == 1:
        return frame.apply(lambda x: (x/(base) - 1) * 100)
    f = frame.apply(lambda x: x.div(base).subtract(1).mul(100), axis=1)
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

def fit_linear_cv(deg, x, y, regularization_factor):
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    poly_features = poly.fit_transform(x.reshape(-1, 1))
    lin_reg = LinearRegression()
    scores = cross_validate(lin_reg, poly_features, y, cv=5, scoring=('neg_mean_squared_error'))
    mses = np.array(scores['test_score'])
    mses -= regularization_factor * deg
    return np.mean(mses)

def fit_linear_pred(deg, before_x, before_y, after_x):
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    poly_features = poly.fit_transform(before_x.reshape(-1, 1))
    lin_reg = LinearRegression()
    lin_reg.fit(poly_features, before_y)
    poly_after = poly.fit_transform(after_x.reshape(-1, 1))
    return lin_reg.predict(poly_after)

def fit_svm_pred(before_x, before_y, after_x):
    reg = SVR(kernel='rbf', C=1.0)
    reg.fit(before_x.reshape(-1, 1), before_y)
    return reg.predict(after_x.reshape(-1, 1))

def fit_and_predict_svm(industry_before, industry_after, company_before, company_after):
    industry_before_x = np.array([int(row) for row in industry_before.index])
    industry_last_index = int(industry_before.index[-1])
    industry_after_x = np.array([int(row)+industry_last_index+1 for row in industry_after.index])
    industry_y = industry_before.to_numpy()
    company_before_x = np.array([int(row) for row in company_before.index])
    company_last_index = int(company_before.index[-1])
    company_after_x = np.array([int(row)+company_last_index+1 for row in company_after.index])
    company_y = company_before.to_numpy()
    industry_pred = fit_svm_pred(industry_before_x, industry_y, industry_after_x)
    company_pred = fit_svm_pred(company_before_x, company_y, company_after_x)
    return industry_pred, company_pred

def fit_rf_cv(depth, x, y, regularization_factor):
    reg = RandomForestRegressor(max_depth=depth, ccp_alpha=regularization_factor)
    scores = cross_validate(reg, x.reshape(-1, 1), y.ravel(), cv=5, scoring=('neg_mean_squared_error'))
    mses = np.array(scores['test_score'])
    return np.mean(mses)

def fit_rf_pred(depth, before_x, before_y, after_x):
    reg = RandomForestRegressor(max_depth=depth)
    reg.fit(before_x.reshape(-1, 1), before_y.ravel())
    return reg.predict(after_x.reshape(-1, 1))

def fit_and_predict_rf(industry_before, industry_after, company_before, company_after):
    depths = [3,4,5,6,7]
    regularization_factor = 0.05
    industry_before_x = np.array([int(row) for row in industry_before.index])
    industry_last_index = int(industry_before.index[-1])
    industry_after_x = np.array([int(row)+industry_last_index+1 for row in industry_after.index])
    industry_y = industry_before.to_numpy()
    industry_means = []
    company_before_x = np.array([int(row) for row in company_before.index])
    company_last_index = int(company_before.index[-1])
    company_after_x = np.array([int(row)+company_last_index+1 for row in company_after.index])
    company_y = company_before.to_numpy()
    company_means = []
    for d in depths:
        industry_mean = fit_rf_cv(d, industry_before_x, industry_y, regularization_factor)
        company_mean = fit_rf_cv(d, company_before_x, company_y, regularization_factor)
        industry_means.append(industry_mean)
        company_means.append(company_mean)
    industry_best_deg = depths[np.argmin(np.array(industry_means))]
    company_best_deg = depths[np.argmin(np.array(company_means))]
    industry_pred = fit_rf_pred(industry_best_deg, industry_before_x, industry_y, industry_after_x)
    company_pred = fit_rf_pred(company_best_deg, company_before_x, company_y, company_after_x)
    return industry_pred, company_pred

def fit_and_predict(industry_before, industry_after, company_before, company_after):
    poly_degrees = [1,2,3,4,5]
    regularization_factor = 1
    industry_before_x = np.array([int(row) for row in industry_before.index])
    industry_last_index = int(industry_before.index[-1])
    industry_after_x = np.array([int(row)+industry_last_index+1 for row in industry_after.index])
    industry_y = industry_before.to_numpy()
    industry_means = []
    company_before_x = np.array([int(row) for row in company_before.index])
    company_last_index = int(company_before.index[-1])
    company_after_x = np.array([int(row)+company_last_index+1 for row in company_after.index])
    company_y = company_before.to_numpy()
    company_means = []
    for d in poly_degrees:
        industry_mean = fit_linear_cv(d, industry_before_x, industry_y, regularization_factor)
        company_mean = fit_linear_cv(d, company_before_x, company_y, regularization_factor)
        industry_means.append(industry_mean)
        company_means.append(company_mean)
    industry_best_deg = poly_degrees[np.argmin(np.array(industry_means))]
    company_best_deg = poly_degrees[np.argmin(np.array(company_means))]
    industry_pred = fit_linear_pred(industry_best_deg, industry_before_x, industry_y, industry_after_x)
    company_pred = fit_linear_pred(company_best_deg, company_before_x, company_y, company_after_x)
    return industry_pred, company_pred

def fit_and_predict_inference(industry_before, industry_after, company_before, company_after):
    poly_degrees = [1,2,3,4,5]
    regularization_factor = 1
    industry_before_x = np.array([int(row) for row in industry_before.index])
    industry_last_index = int(industry_before.index[-1])
    industry_after_x = np.array([int(row)+industry_last_index+1 for row in industry_after.index])
    industry_x = np.concatenate((industry_before_x, industry_after_x))
    industry_y = np.concatenate((industry_before.to_numpy(), industry_after.to_numpy()))
    industry_means = []
    company_before_x = np.array([int(row) for row in company_before.index])
    company_last_index = int(company_before.index[-1])
    company_after_x = np.array([int(row)+company_last_index+1 for row in company_after.index])
    company_x = np.concatenate((company_before_x, company_after_x))
    company_y = np.concatenate((company_before.to_numpy(), company_after.to_numpy()))
    company_means = []
    for d in poly_degrees:
        industry_mean = fit_linear_cv(d, industry_x, industry_y, regularization_factor)
        company_mean = fit_linear_cv(d, company_x, company_y, regularization_factor)
        industry_means.append(industry_mean)
        company_means.append(company_mean)
    industry_best_deg = poly_degrees[np.argmin(np.array(industry_means))]
    company_best_deg = poly_degrees[np.argmin(np.array(company_means))]
    industry_pred = fit_linear_pred(industry_best_deg, industry_x, industry_y, industry_after_x)
    company_pred = fit_linear_pred(company_best_deg, company_x, company_y, company_after_x)
    return industry_pred, company_pred

def score_relative_to_preds(industry_series, company_series, industry_pred_series, company_pred_series):
    industry_scaler = StandardScaler()
    industry = industry_series.to_numpy()
    industry_scaler.fit(industry)

    company_scaler = StandardScaler()
    company = company_series.to_numpy()
    company_scaler.fit(company)
    
    industry_preds = np.array(industry_pred_series).reshape(-1, 1)
    company_preds = np.array(company_pred_series).reshape(-1, 1)
    
    # pos = good, neg = bad
    industry_diff = industry_scaler.transform(industry) - industry_scaler.transform(industry_preds)
    company_diff = company_scaler.transform(company) - company_scaler.transform(company_preds)
    
    industry_me = np.mean(industry_diff)
    company_me = np.mean(company_diff)
    
    return (company_me - industry_me)/abs(industry_me)

def get_likelihood(pct):
    count = 0
    for _, row in companies_df.iterrows():
        if row.loc["FemaleCEO"] == "yes":
            count += 1
    pct_female_ceos = count/companies_df.shape[0]
    total = 0
    #print(count)
    for i in range(count):
        total += math.comb(companies_df.shape[0], i+1)*(pct**(i+1))*((1-pct)**(companies_df.shape[0]-i+1))
    return total

if __name__ == '__main__':
    # Test
    pos = 0
    total = 0
    for name, _ in ceos_df.iterrows():
        print(name)
        a, b, c, d = get_continuous_prices_by_ceo_name(name)
        if not isinstance(a, pd.DataFrame):
            continue
        industry_preds, company_preds = fit_and_predict_rf(a, b, c, d)
        score = score_relative_to_preds(b, d, industry_preds, company_preds)
        total += 1
        if score > 0:
            pos += 1
        print(f"Positive rate: {pos/total}")
    print(f"Likelihood: {get_likelihood(pos/total)}")

    # a, b, c, d = get_continuous_prices_by_ceo_name("Karen S. Lynch")
    # if not isinstance(a, pd.DataFrame):
    #     sys.exit(1)
    # industry_preds, company_preds = fit_and_predict(a, b, c, d)
    # score = score_relative_to_preds(b, d, industry_preds, company_preds)
    # print(score)
