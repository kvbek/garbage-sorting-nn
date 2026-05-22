import os
import pandas as pd

def save_training_history(history, model_type):    
    # Create the logs directory if it doesn't exist yet
    os.makedirs('logs', exist_ok=True)

    # Convert the history dictionary to a Pandas DataFrame
    history_df = pd.DataFrame(history.history)

    # Save it as a CSV (index=False prevents adding an extra column of row numbers)
    history_df.to_csv(f'logs/{model_type}_history.csv', index=False)

    print(f"Training history saved to logs/{model_type}_history.csv")