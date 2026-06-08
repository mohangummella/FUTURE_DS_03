import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================================
# CREATE IMAGES FOLDER
# =====================================

os.makedirs("IMAGES", exist_ok=True)

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv(
    r"DATA\E-commerce Customer Behavior - Sheet1.csv"
)

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns)

# =====================================
# DATASET INFO
# =====================================

print("\nDataset Information")
print(df.info())

# =====================================
# KPI METRICS
# =====================================

total_customers = df["Customer ID"].nunique()
total_revenue = df["Total Spend"].sum()
average_spend = df["Total Spend"].mean()
average_rating = df["Average Rating"].mean()

print("\n========== KPI METRICS ==========")

print("Total Customers:", total_customers)
print("Total Revenue:", round(total_revenue, 2))
print("Average Spend:", round(average_spend, 2))
print("Average Rating:", round(average_rating, 2))

# =====================================
# GENDER DISTRIBUTION
# =====================================

gender_counts = df["Gender"].value_counts()

plt.figure(figsize=(6, 6))

gender_counts.plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Gender Distribution")
plt.ylabel("")

plt.tight_layout()

plt.savefig("IMAGES/gender_distribution.png")

plt.show()

# =====================================
# MEMBERSHIP TYPE ANALYSIS
# =====================================

membership_sales = (
    df.groupby("Membership Type")["Total Spend"]
    .sum()
    .sort_values(ascending=False)
)

print("\nMembership Revenue")
print(membership_sales)

membership_sales.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Revenue by Membership Type")
plt.xlabel("Membership Type")
plt.ylabel("Revenue")

plt.tight_layout()

plt.savefig("IMAGES/membership_revenue.png")

plt.show()

# =====================================
# CITY ANALYSIS
# =====================================

city_sales = (
    df.groupby("City")["Total Spend"]
    .sum()
    .sort_values(ascending=False)
)

print("\nRevenue by City")
print(city_sales)

city_sales.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Revenue by City")
plt.xlabel("City")
plt.ylabel("Revenue")

plt.tight_layout()

plt.savefig("IMAGES/city_revenue.png")

plt.show()

# =====================================
# SATISFACTION ANALYSIS
# =====================================

satisfaction = (
    df["Satisfaction Level"]
    .value_counts()
)

plt.figure(figsize=(8, 5))

sns.barplot(
    x=satisfaction.index,
    y=satisfaction.values
)

plt.title("Customer Satisfaction Level")
plt.xlabel("Satisfaction")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("IMAGES/satisfaction_analysis.png")

plt.show()

# =====================================
# AGE DISTRIBUTION
# =====================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Age"],
    bins=15,
    kde=True
)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("IMAGES/age_distribution.png")

plt.show()

# =====================================
# DISCOUNT IMPACT
# =====================================

discount_sales = (
    df.groupby("Discount Applied")["Total Spend"]
    .sum()
)

print("\nDiscount Analysis")
print(discount_sales)

discount_sales.plot(
    kind="bar",
    figsize=(6, 4)
)

plt.title("Revenue by Discount Applied")
plt.xlabel("Discount Applied")
plt.ylabel("Revenue")

plt.tight_layout()

plt.savefig("IMAGES/discount_analysis.png")

plt.show()

# =====================================
# TOP 10 CUSTOMERS
# =====================================

top_customers = (
    df.sort_values(
        by="Total Spend",
        ascending=False
    )
    .head(10)
)

print("\nTop 10 Customers")
print(
    top_customers[
        ["Customer ID", "Total Spend"]
    ]
)

# =====================================
# PURCHASE FREQUENCY ANALYSIS
# =====================================

purchase_freq = (
    df.groupby("Purchase Frequency")
    .size()
)

purchase_freq.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Purchase Frequency Distribution")
plt.xlabel("Purchase Frequency")
plt.ylabel("Customers")

plt.tight_layout()

plt.savefig("IMAGES/purchase_frequency.png")

plt.show()

# =====================================
# CORRELATION HEATMAP
# =====================================

plt.figure(figsize=(10, 6))

sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("IMAGES/correlation_heatmap.png")

plt.show()

# =====================================
# SPENDING VS RATING
# =====================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Total Spend",
    y="Average Rating"
)

plt.title("Total Spend vs Average Rating")

plt.tight_layout()

plt.savefig("IMAGES/spend_vs_rating.png")

plt.show()

# =====================================
# FINAL BUSINESS INSIGHTS
# =====================================

print("\n========== BUSINESS INSIGHTS ==========")

print("1. Membership type influences customer spending.")
print("2. High-spending cities contribute major revenue.")
print("3. Customer satisfaction impacts retention.")
print("4. Discounts affect purchasing behavior.")
print("5. Top customers can be targeted for loyalty programs.")
print("6. Purchase frequency helps identify valuable customers.")

print("\nAnalysis Completed Successfully.")