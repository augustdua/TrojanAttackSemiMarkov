import pandas as pd

def save_log_to_csv(log, filename="data/trojan_sim_log.csv"):
    """Saves simulation log to CSV."""
    df = pd.DataFrame(log)
    df.to_csv(filename, index=False)
    print(f"Log saved to {filename}")
