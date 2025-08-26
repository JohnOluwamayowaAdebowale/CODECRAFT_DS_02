import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "train.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_titanic(df):
    # Basic cleaning & feature engineering
    df = df.copy()
    # Fill Age with median by Pclass & Sex to be smarter
    df["Age"] = df.groupby(["Pclass","Sex"])["Age"].transform(lambda s: s.fillna(s.median()))
    # Fill Embarked with mode
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])
    # Extract Title
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    # Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    # IsAlone
    df["IsAlone"] = (df["FamilySize"]==1).astype(int)
    # Ticket prefix
    df["TicketPrefix"] = df["Ticket"].str.replace(r"[^A-Za-z]+", "", regex=True).replace("", "NONE")
    # Cabin present
    df["HasCabin"] = (~df["Cabin"].isna()).astype(int)
    return df

def plot_counts(df, col, title, fname):
    plt.figure()
    counts = df[col].value_counts(dropna=False).sort_index()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname))
    plt.close()

def plot_survival_rate(df, by, fname):
    rates = df.groupby(by)["Survived"].mean().sort_values(ascending=False)
    plt.figure()
    plt.bar(rates.index.astype(str), rates.values)
    plt.title(f"Survival Rate by {by}")
    plt.xlabel(by)
    plt.ylabel("Survival Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname))
    plt.close()

def main():
    df = pd.read_csv(DATA_PATH)
    print("Raw shape:", df.shape)

    df_clean = clean_titanic(df)
    print("Cleaned shape:", df_clean.shape)

    # Save cleaned
    df_clean.to_csv(os.path.join(OUTPUT_DIR, "titanic_cleaned.csv"), index=False)

    # Basic distributions
    plot_counts(df_clean, "Pclass", "Passenger Class Distribution", "pclass_dist.png")
    plot_counts(df_clean, "Sex", "Sex Distribution", "sex_dist.png")
    plot_counts(df_clean, "Embarked", "Embarked Distribution", "embarked_dist.png")

    # Survival rates
    plot_survival_rate(df_clean, "Pclass", "survival_by_pclass.png")
    plot_survival_rate(df_clean, "Sex", "survival_by_sex.png")
    plot_survival_rate(df_clean, "Embarked", "survival_by_embarked.png")
    plot_survival_rate(df_clean, "IsAlone", "survival_by_isalone.png")

    # Age histogram
    plt.figure()
    plt.hist(df_clean["Age"].dropna(), bins=30)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "age_hist.png"))
    plt.close()

    print("EDA complete. See outputs in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
