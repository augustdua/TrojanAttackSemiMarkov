```markdown
# Sequential Cyber-Attack Simulator: A Semi-Markov Process Approach

## Overview

This project implements a semi-Markov process simulation framework for modeling and analyzing sequential cyber-attacks, based on the research paper:

> **Probabilistic modeling and analysis of sequential cyber-attacks**  
> by Qisi Liu, Liudong Xing, and Chencheng Zhou  
> *Research Article*

The simulator models the progression of trojan attacks through different states (Clean, Acquisition, Infected, and Fraud) using a semi-Markov process. This approach provides a more realistic representation of cyber-attack dynamics compared to traditional Markov models by incorporating variable sojourn times in each state.

## Research Foundation

This implementation is directly based on the methodologies proposed by Liu, Xing, and Zhou, who introduced a probabilistic framework for analyzing sequential cyber-attacks. Their work demonstrated that:

1. The temporal evolution of cyber-attacks can be effectively modeled using semi-Markov processes
2. Variable sojourn times better represent real-world attack dynamics than constant rates
3. Steady-state analysis provides valuable insights into long-term system security

Our implementation extends their theoretical framework into a practical simulation tool that enables security researchers to:
- Visualize attack progression through different states
- Analyze state distributions at multiple time checkpoints
- Calculate steady-state distributions
- Examine transition probabilities between attack states

## Key Features

- **Semi-Markov Simulation**: Implementation of a semi-Markov model for sequential cyber-attacks
- **State Transition Modeling**: Configurable transition matrix between system states
- **Sojourn Time Distributions**: Weibull distributions for modeling time spent in each state
- **Checkpoint Analysis**: State distribution analysis at predefined time points (1s, 1m, 15m, 1h, 1d, 1mo)
- **Steady State Calculation**: Convergence-based calculation of long-term state probabilities
- **Interactive Visualizations**:
  - State transition diagram with probability-weighted transitions
  - State distribution over time charts
  - Checkpoint distribution cards
  - Sojourn time analysis
  - Transition matrix heatmap

## Repository Structure

├── data/
│   └── trojan_sim_log.csv       # Sample simulation log data
├── src/
│   ├── config.py                # Configuration parameters and states
│   ├── sojourn.py               # Sojourn time generation functions
│   ├── semi_markov_simulator.py # Core simulation engine
│   ├── semi_markov_dashboard.py # Visualization dashboard
│   ├── logger.py                # Logging utilities
│   ├── visualizer.py            # Visualization components
│   └── main.py                  # Main execution script
├── tests/
│   └── ...                      # Test scripts
├── torjan_attack_crq_sim.ipynb  # Jupyter notebook with interactive visualizations
└── requirements.txt             # Project dependencies
```

## Mathematical Model

The simulator implements a semi-Markov process with:

- **State Space**: S = {Clean, Acquisition, Infected, Fraud}
- **Transition Matrix**: P = [p_ij] where p_ij represents the probability of transitioning from state i to state j
- **Sojourn Times**: Random variables with Weibull distributions parameterized by shape (k) and scale (λ) parameters
- **Steady State Distribution**: π = [π_Clean, π_Acquisition, π_Infected, π_Fraud]

The simulation calculates the probability distribution across states at any time t, and determines the steady-state distribution as t → ∞.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Jupyter Notebook

The easiest way to interact with the simulator is through the Jupyter notebook:

```bash
jupyter notebook torjan_attack_crq_sim.ipynb
```

This will open an interactive environment where you can:
- Run simulations with different parameters
- View visualizations of attack progression
- Analyze state distributions at various checkpoints
- Examine the steady-state distribution

### Using the Python API

You can also use the simulator programmatically:

```python
from src.semi_markov_simulator import SemiMarkovSimulator
from src.semi_markov_dashboard import SemiMarkovAnalyzer

# Create and run simulator
simulator = SemiMarkovSimulator()
simulator.run(max_iterations=50000, tolerance=0.001)

# Visualize results
analyzer = SemiMarkovAnalyzer()
analyzer.simulator = simulator
analyzer.logs = pd.DataFrame(simulator.log)

# Generate visualizations
analyzer.plot_state_diagram()
analyzer.plot_state_distribution_over_time()
analyzer.display_checkpoint_cards()
```

## Customization

### Modifying the Transition Matrix

You can adjust the transition probabilities between states by modifying the `TRANSITION_MATRIX` in `config.py`:

```python
TRANSITION_MATRIX = np.array([
    [0.7, 0.3, 0.0, 0.0],  # From Clean
    [0.4, 0.3, 0.3, 0.0],  # From Acquisition
    [0.1, 0.3, 0.4, 0.2],  # From Infected
    [0.0, 0.0, 0.3, 0.7],  # From Fraud
])
```

### Adjusting Sojourn Time Parameters

Sojourn time parameters can be modified through the `SemiMarkovSimulator` class:

```python
simulator = SemiMarkovSimulator()
simulator.update_sojourn_params({
    'Clean': {'shape': 1.5, 'scale': 10.0},
    'Acquisition': {'shape': 1.2, 'scale': 5.0},
    'Infected': {'shape': 1.0, 'scale': 3.0},
    'Fraud': {'shape': 0.8, 'scale': 2.0}
})
```

## Advanced Analysis

### Cyber Risk Quantification

The simulator can be used for cyber risk quantification by:
1. Determining the probability of being in the "Fraud" state
2. Calculating the expected financial impact based on the average cost per fraudulent incident
3. Analyzing how changes in security measures (represented by transition probabilities) affect overall risk

### Security Control Effectiveness

By modifying transition probabilities, you can model the effectiveness of security controls:
- Reduced Clean→Acquisition probability represents improved perimeter defenses
- Reduced Acquisition→Infected probability models better detection capabilities
- Increased Infected→Clean probability represents effective remediation processes

## Results Interpretation

The key outputs to focus on when analyzing results:

1. **Steady-State Distribution**: Shows the long-term probability of being in each state, representing the system's equilibrium behavior
2. **Time to Steady State**: Indicates how quickly the system reaches its equilibrium, which relates to the speed of attack propagation
3. **State Distribution Over Time**: Reveals the temporal dynamics of the attack process and potential bottlenecks
4. **Sojourn Time Analysis**: Provides insights into how long systems remain in vulnerable states

## Contributing

Contributions to this project are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citing

If you use this simulator in your research, please cite both this repository and the original paper:

```
Liu, Q., Xing, L., & Zhou, C. (YEAR). Probabilistic modeling and analysis of sequential cyber-attacks. JOURNAL/CONFERENCE NAME, VOLUME(ISSUE), PAGES.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Qisi Liu, Liudong Xing, and Chencheng Zhou for their foundational research
- The open-source community for the various libraries used in this project
```
