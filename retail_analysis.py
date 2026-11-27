# Retail Business Performance & Profitability Analysis
# Run: pip install pandas seaborn matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("retail_transactions.csv", parse_dates=["order_date"])
df["region"] = df["region"].fillna("Unknown")
df["discount_rate"] = df["discount_rate"].fillna(0)
df["inventory_days"] = df["inventory_days"].fillna(df["inventory_days"].median())
df["profit_margin_pct"] = 100 * df["profit"] / df["revenue"].replace(0, pd.NA)

category_summary = df.groupby("category").agg(
    revenue=("revenue", "sum"), profit=("profit", "sum"),
    avg_inventory_days=("inventory_days", "mean"),
    units_sold=("quantity", "sum")
)
category_summary["profit_margin_pct"] = 100 * category_summary["profit"] / category_summary["revenue"]
print(category_summary.sort_values("profit_margin_pct"))

corr = df[["inventory_days", "profit_margin_pct"]].dropna().corr(method="pearson").iloc[0, 1]
print(f"Pearson correlation (inventory days vs profit margin): {corr:.3f}")

sns.regplot(data=df, x="inventory_days", y="profit_margin_pct",
            scatter_kws={"alpha": 0.35}, line_kws={"color": "red"})
plt.title(f"Inventory Days vs Profit Margin (r={corr:.2f})")
plt.tight_layout()
plt.savefig("inventory_profit_correlation.png", dpi=150)
plt.show()

seasonal = df.groupby(["season", "category"])[["revenue", "profit"]].sum().reset_index()
print(seasonal)
