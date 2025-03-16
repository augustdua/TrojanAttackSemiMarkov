import numpy as np
from config import STATES, TRANSITION_MATRIX
from sojourn import generate_sojourn_time

class SemiMarkovSimulator:
    def __init__(self):
        self.current_state = "Clean"  # Start in the Clean state
        self.time = 0
        self.state_counts = {state: 0 for state in STATES}
        self.log = []
        self.checkpoints = [1, 60, 900, 3600, 86400, 2.63e6]  # 1s, 1m, 15m, 1h, 1d, 1mo
        self.checkpoint_results = {time: {state: 0 for state in STATES} for time in self.checkpoints}
        self.reached_checkpoints = set()  # To track which checkpoints we've passed
        self.steady_state_time = None  # To track when steady state is reached

    def run(self, max_iterations=50000, tolerance=0.001, check_interval=1000):
        prev_steady_state_probs = {state: 0 for state in STATES}

        for iteration in range(max_iterations):
            sojourn_time = generate_sojourn_time(self.current_state)
            self.time += sojourn_time
            
            # Store current state before transition
            current_state = self.current_state

            # Choose next state
            next_state = np.random.choice(STATES, p=TRANSITION_MATRIX[STATES.index(self.current_state)])
            self.state_counts[next_state] += 1
            
            # Create a single complete log entry
            self.log.append({
                "Iteration": iteration,
                "Time": self.time,
                "State": current_state,
                "Sojourn Time": sojourn_time,
                "Transition From": current_state,
                "Transition To": next_state
            })
            
            # Update current state
            self.current_state = next_state

            # Record state distribution at checkpoints and print status
            for checkpoint in self.checkpoints:
                if self.time >= checkpoint and checkpoint not in self.reached_checkpoints:
                    total_time_spent = sum(self.state_counts.values()) + 1e-6  # Avoid division by zero
                    for state in STATES:
                        self.checkpoint_results[checkpoint][state] = self.state_counts[state] / total_time_spent
                    
                    # Print checkpoint information
                    time_label = self.get_time_label(checkpoint)
                    print(f"Reached {time_label} checkpoint at iteration {iteration}, time {self.time:.2f}")
                    print(f"State distribution: {self.checkpoint_results[checkpoint]}")
                    
                    self.reached_checkpoints.add(checkpoint)

            # Check for steady state
            if iteration % check_interval == 0 and iteration > 0:
                steady_state_probs = {state: self.state_counts[state] / sum(self.state_counts.values()) for state in STATES}
                diff = max(abs(steady_state_probs[s] - prev_steady_state_probs[s]) for s in STATES)
                
                print(f"Iteration {iteration}: Current steady state difference: {diff:.6f}")

                if diff < tolerance:
                    self.steady_state_time = self.time
                    print(f"Converged after {iteration} iterations at time {self.steady_state_time:.2f}!")
                    print(f"Simulated time to reach steady state: {self.format_time(self.steady_state_time)}")
                    print(f"Final state distribution: {steady_state_probs}")
                    self.checkpoint_results["Steady-State"] = steady_state_probs
                    break

                prev_steady_state_probs = steady_state_probs.copy()
        
        # Print summary if we completed all iterations without convergence
        if iteration == max_iterations - 1:
            print(f"Reached maximum iterations ({max_iterations}) without convergence")
            print(f"Total simulation time: {self.format_time(self.time)}")
            steady_state_probs = {state: self.state_counts[state] / sum(self.state_counts.values()) for state in STATES}
            print(f"Final state distribution: {steady_state_probs}")
            self.checkpoint_results["Final"] = steady_state_probs
    
    def get_time_label(self, seconds):
        """Convert seconds to a human-readable label"""
        if seconds == 1:
            return "1 second"
        elif seconds == 60:
            return "1 minute"
        elif seconds == 900:
            return "15 minutes"
        elif seconds == 3600:
            return "1 hour"
        elif seconds == 86400:
            return "1 day"
        elif seconds == 2.63e6:
            return "1 month"
        else:
            return f"{seconds} seconds"
    
    def format_time(self, seconds):
        """Format time in seconds to a human-readable string"""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f} minutes"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.2f} hours"
        elif seconds < 30 * 86400:
            days = seconds / 86400
            return f"{days:.2f} days"
        else:
            months = seconds / (30 * 86400)
            return f"{months:.2f} months"
    
    def export_log(self, filename="simulation_log.csv"):
        """Export the log to a CSV file"""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ["Iteration", "Time", "State", "Sojourn Time", "Transition From", "Transition To"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for entry in self.log:
                writer.writerow(entry)
            
        print(f"Log exported to {filename}")
        
        # Include steady state time in the summary
        if self.steady_state_time:
            print(f"Steady state was reached at {self.format_time(self.steady_state_time)}")