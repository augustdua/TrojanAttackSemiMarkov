from semi_markov_simulator import SemiMarkovSimulator
from logger import save_log_to_csv
from visualizer import plot_steady_state_distribution

if __name__ == "__main__":
    simulator = SemiMarkovSimulator()
    simulator.run()

    # Save log
    save_log_to_csv(simulator.log)

    # Compute and visualize steady-state probabilities
    steady_state_probs = {state: simulator.state_counts[state] / sum(simulator.state_counts.values()) for state in simulator.state_counts}
    plot_steady_state_distribution(steady_state_probs)
