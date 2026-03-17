import random
import numpy as np
import pandas as pd
from datasets import load_dataset

def simulate_data_drift(test_df):      # corrupting 40% of texts with slangs/abbreviations
    drift_df = test_df.copy()
    idx = test_df.sample(frac=0.40, random_state=42).index

    def replace_words(text):
        if "card" in text:
            text = text.replace("card", random.choice(["ard","crd","crad","catd"]))
        if "account" in text:
            text = text.replace("account", random.choice(["acc","a/c","acount"]))
        if "transfer" in text:
            text = text.replace("transfer", random.choice(["transfr","tranfer","acount"]))
        if "amount" in text:
            text = text.replace("amount", random.choice(["amt","amout","amnt"]))
        if "please" in text:
            text = text.replace("please", "pls")
        if "cannot" in text:
            text = text.replace("cannot", "cant")
        return text

    drift_df.loc[idx, "text"] = test_df.loc[idx, "text"].apply(replace_words)
    return drift_df


def simulate_label_drift(test_df):     # oversample 5-6 intents to make up 70% of the data.
    drift_df = test_df.copy()
    chosen_intents = [4, 9, 54, 67]    # randomly selected 4 intents

    dominant = test_df[test_df['label'].isin(chosen_intents)]
    oversampled = pd.concat([dominant] * 6).reset_index(drop=True)
    rest = test_df[~test_df['label'].isin(chosen_intents)]
    drift_df = (pd.concat([oversampled, rest]).reset_index(drop=True)).sample(frac = 1, random_state=42)
    return drift_df


def simulate_concept_drift(test_df):   # shuffle, reassign labels of 40% text
    drift_df = test_df.copy()
    idx = test_df.sample(frac=0.40, random_state=42).index

    labels = test_df.loc[idx, 'label'].values.copy()
    np.random.shuffle(labels)          # it returns None, so shuffling in place
    drift_df.loc[idx, 'label'] = labels

    return drift_df


def main():  # loading the test split and calling all 3 above functions, then saving results to parquet
    dataset = load_dataset("banking77")
    test_df = dataset['test'].to_pandas()

    data_drift = simulate_data_drift(test_df)
    data_drift.to_parquet("data/production/data_drift.parquet", index=False)    # to_parquet returns None
    label_drift = simulate_label_drift(test_df)
    label_drift.to_parquet("data/production/label_drift.parquet", index=False)
    concept_drift = simulate_concept_drift(test_df)
    concept_drift.to_parquet("data/production/concept_drift.parquet", index=False)

if __name__ == "__main__":
    main()