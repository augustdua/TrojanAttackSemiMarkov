import matplotlib.pyplot as plt
import seaborn as sns

def plot_steady_state_distribution(steady_state_probs):
    """Plots steady-state probabilities as a bar chart."""
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(steady_state_probs.keys()), y=list(steady_state_probs.values()), palette="coolwarm")
    plt.xlabel("State")
    plt.ylabel("Steady-State Probability")
    plt.title("Steady-State Distribution of Trojan Attack States")
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.show()
