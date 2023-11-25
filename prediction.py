from helpers.helpers import *
import numpy as np
import pandas as pd
import yfinance as yf

ceos_df = pd.read_csv("CEOS_Kaggle.csv")
ceos_df.set_index("CEO", inplace=True)
companies_df = pd.read_csv("fortune1000_2023.csv")
companies_df = companies_df[companies_df['Ticker'].notna()]
companies_df.set_index("Ticker", inplace=True)

ceo_names = ceos_df.index.tolist()
dfs = []
dfs_inference = []


#   # fit and predict
#   for name in ceo_names:
#     a, b, c, d = get_continuous_prices_by_ceo_name(name)
#     industry_preds, company_preds = fit_and_predict(a, b, c, d)
#     if not isinstance(a, pd.DataFrame):
#             continue
#     df_temp_1 = pd.DataFrame({
#           "name": [name] * len(np.concatenate(industry_preds)),
#           "val": ["Company"] * len(np.concatenate(industry_preds)),
#           "status": ["After"] * len(np.concatenate(industry_preds)),
#           "Value": np.concatenate(company_preds),
#           "Day": range(1, len(np.concatenate(industry_preds)) + 1)
#       })
#     df_temp_2 = pd.DataFrame({
#           "name": [name] * len(np.concatenate(industry_preds)),
#           "val": ["Industry Avg"] * len(np.concatenate(industry_preds)),
#           "status": ["After"] * len(np.concatenate(industry_preds)),
#           "Value": np.concatenate(industry_preds),
#           "Day": range(1, len(np.concatenate(industry_preds)) + 1)
#       })
#     dfs.append(df_temp_1)
#     dfs.append(df_temp_2)
# 
#   # fit and predict inference
#   for name in ceo_names:
#     a, b, c, d = get_continuous_prices_by_ceo_name(name)
#     if not isinstance(a, pd.DataFrame):
#             continue
#     industry_preds, company_preds = fit_and_predict_inference(a, b, c, d)
#     df_temp_1 = pd.DataFrame({
#           "name": [name] * len(np.concatenate(industry_preds)),
#           "val": ["Company"] * len(np.concatenate(industry_preds)),
#           "status": ["After"] * len(np.concatenate(industry_preds)),
#           "Value": np.concatenate(company_preds),
#           "Day": range(1, len(np.concatenate(industry_preds)) + 1)
#       })
#     df_temp_2 = pd.DataFrame({
#           "name": [name] * len(np.concatenate(industry_preds)),
#           "val": ["Industry Avg"] * len(np.concatenate(industry_preds)),
#           "status": ["After"] * len(np.concatenate(industry_preds)),
#           "Value": np.concatenate(industry_preds),
#           "Day": range(1, len(np.concatenate(industry_preds)) + 1)
#       })
#     dfs_inference.append(df_temp_1)
#     dfs_inference.append(df_temp_2)
# 
# fit_and_predict_results = pd.concat(dfs, ignore_index=True)
# fit_and_predict_inference_results = pd.concat(dfs_inference, ignore_index=True)
