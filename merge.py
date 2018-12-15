xgb_d = pd.read_csv("results/xgb.csv", index_col=0)
lgb_d = pd.read_csv("results/lgb.csv", index_col=0)
((xgb_d + lgb_d) / 2).to_csv("1.csv")