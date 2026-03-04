# Import libraries
import pandas as pd

# Balance dataset for steering
def balance_steering(df):
    """
    Balance steering data to turn imbalanced dataset into balanced.
    :param df: Dataframe containing the data
    :return: balanced_df
    """
    # Separate steering by category
    straight = df[abs(df.steering) < 0.05]
    mild = df[(abs(df.steering) >= 0.05) & (abs(df.steering) < 0.3)]
    strong = df[abs(df.steering) >= 0.3]

    # Sample a fraction of the straight driving
    target = len(mild) + len(strong)
    if len(straight) > target and target > 0:
        straight = straight.sample(n=len(mild) + len(strong), random_state=42)

    # Balance dataframe
    balanced_df = pd.concat([straight, mild, strong])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df