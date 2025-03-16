# Semi-Markov Process Simulation and Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import networkx as nx
from matplotlib.colors import to_rgba
import io
import base64
from matplotlib.animation import FuncAnimation

# Import your existing simulation code
from config import STATES, TRANSITION_MATRIX
from sojourn import generate_sojourn_time
from semi_markov_simulator import SemiMarkovSimulator

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
colors = {
    'Clean': '#10b981',      # Green
    'Acquisition': '#3b82f6', # Blue
    'Infected': '#f59e0b',    # Orange
    'Fraud': '#ef4444'       # Red
}

class SemiMarkovDashboard:
    def __init__(self):
        self.simulator = SemiMarkovSimulator()
        self.states = STATES
        self.init_transition_matrix = np.array(TRANSITION_MATRIX)
        self.checkpoints = [1, 60, 900, 3600, 86400, 2.63e6]  # 1s, 1m, 15m, 1h, 1d, 1mo
        self.checkpoint_labels = ["1 sec", "1 min", "15 min", "1 hour", "1 day", "1 month"]
        self.logs = None
        self.create_widgets()
        self.setup_dashboard()
        
    def create_widgets(self):
        # Simulation parameters
        self.max_iterations_input = widgets.IntText(
            value=50000,
            description='Max Iterations:',
            disabled=False
        )
        self.tolerance_input = widgets.FloatText(
            value=0.001,
            description='Tolerance:',
            disabled=False
        )
        
        # Create transition matrix widgets
        self.transition_matrix_widgets = {}
        for i, from_state in enumerate(self.states):
            row_widgets = {}
            for j, to_state in enumerate(self.states):
                row_widgets[to_state] = widgets.FloatSlider(
                    value=self.init_transition_matrix[i][j],
                    min=0,
                    max=1.0,
                    step=0.01,
                    description=f'{from_state} → {to_state}:',
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='.2f',
                )
            self.transition_matrix_widgets[from_state] = row_widgets

        # Create sojourn parameter widgets
        self.sojourn_params_widgets = {}
        for state in self.states:
            self.sojourn_params_widgets[state] = {
                'shape': widgets.FloatText(
                    value=1.5,
                    description=f'{state} Shape:',
                    disabled=False
                ),
                'scale': widgets.FloatText(
                    value=5.0,
                    description=f'{state} Scale:',
                    disabled=False
                )
            }

        # Buttons
        self.run_button = widgets.Button(
            description='Run Simulation',
            disabled=False,
            button_style='success', 
            tooltip='Click to run the simulation',
            icon='play'
        )
        
        # Output area for plots
        self.output = widgets.Output()
        
        # Connect event handlers
        self.run_button.on_click(self.run_simulation)
        
        # Tab for transition matrix normalization
        self.normalize_button = widgets.Button(
            description='Normalize Matrix',
            disabled=False,
            button_style='info',
            tooltip='Normalize rows to sum to 1',
            icon='refresh'
        )
        self.normalize_button.on_click(self.normalize_matrix)
        
    def normalize_matrix(self, button):
        """Normalize each row of the transition matrix to sum to 1."""
        for from_state in self.states:
            # Get current values
            row_widgets = self.transition_matrix_widgets[from_state]
            row_sum = sum(widget.value for widget in row_widgets.values())
            
            if row_sum > 0:
                # Update each slider with normalized value
                for to_state, widget in row_widgets.items():
                    widget.value = widget.value / row_sum
    
    def get_transition_matrix(self):
        """Get the current transition matrix from widgets."""
        matrix = np.zeros((len(self.states), len(self.states)))
        for i, from_state in enumerate(self.states):
            for j, to_state in enumerate(self.states):
                matrix[i][j] = self.transition_matrix_widgets[from_state][to_state].value
        return matrix
    
    def get_sojourn_params(self):
        """Get the current sojourn parameters from widgets."""
        params = {}
        for state in self.states:
            params[state] = {
                'shape': self.sojourn_params_widgets[state]['shape'].value,
                'scale': self.sojourn_params_widgets[state]['scale'].value
            }
        return params
    
    def update_simulator(self):
        """Update the simulator with current parameter values."""
        # Get parameter values from widgets
        transition_matrix = self.get_transition_matrix()
        sojourn_params = self.get_sojourn_params()
        
        # Create a new simulator instance
        self.simulator = SemiMarkovSimulator()
        
        # Update with current parameters
        self.simulator.update_transition_matrix(transition_matrix)
        self.simulator.update_sojourn_params(sojourn_params)
        
        return self.simulator
    
    def run_simulation(self, button):
        """Run the simulation with current parameters."""
        with self.output:
            clear_output(wait=True)
            print("Running simulation...")
            
            # Update simulator with current parameters
            simulator = self.update_simulator()
            
            # Run simulation
            max_iterations = self.max_iterations_input.value
            tolerance = self.tolerance_input.value
            
            try:
                simulator.run(max_iterations=max_iterations, tolerance=tolerance)
                self.logs = pd.DataFrame(simulator.log)
                self.display_results()
            except Exception as e:
                print(f"Error running simulation: {str(e)}")
    
    def display_results(self):
        """Display all visualization results."""
        if self.logs is None or len(self.logs) == 0:
            print("No simulation data available.")
            return
        
        # Create a grid of subplots
        fig = plt.figure(figsize=(18, 24))
        
        # 1. Show checkpoint distributions - Top row
        ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
        self.plot_checkpoint_distributions(ax1)
        
        # 2. Show state transition diagram - Middle left
        ax2 = plt.subplot2grid((4, 2), (1, 0))
        self.plot_state_diagram(ax2)
        
        # 3. Show steady state distribution - Middle right
        ax3 = plt.subplot2grid((4, 2), (1, 1))
        self.plot_steady_state(ax3)
        
        # 4. Show state distribution over time - Bottom left
        ax4 = plt.subplot2grid((4, 2), (2, 0))
        self.plot_state_distribution_over_time(ax4)
        
        # 5. Show sojourn time distribution - Bottom right
        ax5 = plt.subplot2grid((4, 2), (2, 1))
        self.plot_sojourn_time_distribution(ax5)
        
        # 6. Show simulation statistics - Bottom row
        ax6 = plt.subplot2grid((4, 2), (3, 0), colspan=2)
        self.plot_simulation_stats(ax6)
        
        plt.tight_layout()
        plt.show()
        
        # Display the transition probability matrix as a heatmap
        self.plot_transition_matrix()
        
        # Extra: Display the checkpoint cards
        self.display_checkpoint_cards()
    
    def plot_checkpoint_distributions(self, ax):
        """Plot the state distributions at different checkpoints."""
        checkpoint_results = self.simulator.checkpoint_results
        
        # Extract checkpoint data
        data = []
        labels = []
        
        for i, checkpoint in enumerate(self.checkpoints):
            checkpoint_key = checkpoint
            if checkpoint_key in checkpoint_results:
                data.append([checkpoint_results[checkpoint_key].get(state, 0) * 100 for state in self.states])
                labels.append(self.checkpoint_labels[i])
        
        # Add steady state if available
        if "Steady-State" in checkpoint_results:
            data.append([checkpoint_results["Steady-State"].get(state, 0) * 100 for state in self.states])
            labels.append("Steady State")
        
        # Create the plot
        x = np.arange(len(labels))
        width = 0.8 / len(self.states)
        
        for i, state in enumerate(self.states):
            ax.bar(x + i * width, [d[i] for d in data], width, label=state, color=colors.get(state, f'C{i}'))
        
        ax.set_title('State Distribution at Different Time Points')
        ax.set_xlabel('Time Point')
        ax.set_ylabel('Percentage (%)')
        ax.set_xticks(x + width * (len(self.states) - 1) / 2)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True)
    
    def plot_state_diagram(self, ax):
        """Plot the state transition diagram."""
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for state in self.states:
            G.add_node(state)
        
        # Add edges with weights from transition matrix
        transition_matrix = self.get_transition_matrix()
        for i, from_state in enumerate(self.states):
            for j, to_state in enumerate(self.states):
                if transition_matrix[i][j] > 0.01:  # Only add significant transitions
                    G.add_edge(from_state, to_state, weight=transition_matrix[i][j])
        
        # Get steady state distribution for node sizes
        steady_state = self.simulator.checkpoint_results.get("Steady-State", {})
        
        # Set node colors and sizes
        node_colors = [colors.get(state, 'gray') for state in G.nodes()]
        node_sizes = [1000 * (steady_state.get(state, 0.1) + 0.1) for state in G.nodes()]  # Min size of 0.1
        
        # Calculate edge widths based on transition probabilities
        edge_widths = [2 * G[u][v]['weight'] for u, v in G.edges()]
        
        # Position nodes in a circle
        pos = nx.circular_layout(G)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='white', ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray', 
                               connectionstyle='arc3,rad=0.2', arrowsize=15, ax=ax)
        
        # Add edge labels (transition probabilities)
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
        
        ax.set_title('State Transition Diagram')
        ax.axis('off')
    
    def plot_steady_state(self, ax):
        """Plot the steady state distribution."""
        steady_state = self.simulator.checkpoint_results.get("Steady-State", {})
        
        if not steady_state:
            ax.text(0.5, 0.5, "Steady state not reached", 
                    horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
            return
        
        # Get the values in the same order as states
        values = [steady_state.get(state, 0) * 100 for state in self.states]
        
        # Create a horizontal bar chart
        bars = ax.barh(self.states, values, color=[colors.get(state, f'C{i}') for i, state in enumerate(self.states)])
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                    ha='left', va='center')
        
        ax.set_title('Steady State Distribution')
        ax.set_xlabel('Percentage (%)')
        ax.set_xlim(0, 100)
        ax.xaxis.grid(True)
    
    def plot_state_distribution_over_time(self, ax):
        """Plot how the state distribution changes over time."""
        if self.logs is None or len(self.logs) == 0:
            ax.text(0.5, 0.5, "No log data available", 
                    horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
            return
        
        # Process log data - group by time chunks
        max_time = self.logs['Time'].max()
        num_chunks = 20  # Number of time chunks to divide the simulation into
        chunk_size = max_time / num_chunks
        
        # Group by time chunk and count states
        self.logs['TimeChunk'] = (self.logs['Time'] / chunk_size).astype(int)
        state_counts = self.logs.groupby(['TimeChunk', 'State']).size().unstack().fillna(0)
        
        # Convert to percentages
        state_percentages = state_counts.div(state_counts.sum(axis=1), axis=0) * 100
        
        # Plot the data
        time_points = state_percentages.index * chunk_size
        for state in self.states:
            if state in state_percentages.columns:
                ax.plot(time_points, state_percentages[state], 
                        marker='o', markersize=4, 
                        label=state, color=colors.get(state, None))
        
        ax.set_title('State Distribution Over Time')
        ax.set_xlabel('Simulation Time')
        ax.set_ylabel('Percentage (%)')
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(True)
        
        # Format x-axis with time labels
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            elif seconds < 86400:
                return f"{seconds/3600:.1f}h"
            return f"{seconds/86400:.1f}d"
        
        # Set a reasonable number of ticks
        num_ticks = min(10, len(time_points))
        tick_indices = np.linspace(0, len(time_points)-1, num_ticks, dtype=int)
        ax.set_xticks(time_points[tick_indices])
        ax.set_xticklabels([format_time(t) for t in time_points[tick_indices]])
    
    def plot_sojourn_time_distribution(self, ax):
        """Plot the distribution of sojourn times for each state."""
        if self.logs is None or len(self.logs) == 0:
            ax.text(0.5, 0.5, "No log data available", 
                    horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
            return
        
        # Group by state and calculate statistics
        sojourn_stats = self.logs.groupby('State')['Sojourn Time'].agg(['mean', 'min', 'max']).reset_index()
        
        # Sort by mean sojourn time
        sojourn_stats = sojourn_stats.sort_values('mean', ascending=False)
        
        # Create bar plot for mean sojourn time
        bars = ax.bar(sojourn_stats['State'], sojourn_stats['mean'], 
                      color=[colors.get(state, 'gray') for state in sojourn_stats['State']])
        
        # Add error bars for min/max
        ax.errorbar(sojourn_stats['State'], sojourn_stats['mean'], 
                    yerr=[sojourn_stats['mean'] - sojourn_stats['min'], 
                          sojourn_stats['max'] - sojourn_stats['mean']],
                    fmt='none', ecolor='black', capsize=5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, f'{height:.1f}s',
                    ha='center', va='bottom')
        
        ax.set_title('Average Sojourn Time by State')
        ax.set_xlabel('State')
        ax.set_ylabel('Time (seconds)')
        ax.yaxis.grid(True)
    
    def plot_simulation_stats(self, ax):
        """Plot various simulation statistics."""
        if self.logs is None or len(self.logs) == 0:
            ax.text(0.5, 0.5, "No log data available", 
                    horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
            return
        
        # Create a text box with simulation statistics
        max_time = self.logs['Time'].max()
        iterations = len(self.logs)
        state_changes = self.logs.dropna(subset=['Transition From', 'Transition To']).shape[0]
        
        # Calculate convergence time if available
        steady_state_time = getattr(self.simulator, 'steady_state_time', None)
        
        stats_text = [
            f"Total Simulation Time: {self.format_time(max_time)}",
            f"Total Iterations: {iterations}",
            f"State Transitions: {state_changes}",
        ]
        
        if steady_state_time:
            stats_text.append(f"Time to Steady State: {self.format_time(steady_state_time)}")
        
        # Add transition matrix stats
        transition_matrix = self.get_transition_matrix()
        stats_text.append("\nTransition Probabilities:")
        for i, from_state in enumerate(self.states):
            for j, to_state in enumerate(self.states):
                if transition_matrix[i][j] > 0.01:  # Only show significant transitions
                    stats_text.append(f"  {from_state} → {to_state}: {transition_matrix[i][j]:.2f}")
        
        # Add sojourn parameter stats
        sojourn_params = self.get_sojourn_params()
        stats_text.append("\nSojourn Parameters:")
        for state, params in sojourn_params.items():
            stats_text.append(f"  {state}: Shape={params['shape']:.2f}, Scale={params['scale']:.2f}")
        
        ax.text(0.5, 0.5, "\n".join(stats_text), 
                horizontalalignment='center', verticalalignment='center', 
                bbox=dict(boxstyle="round,pad=1", facecolor='#f8f9fa', alpha=0.7))
        ax.axis('off')
        ax.set_title('Simulation Statistics')
    
    def plot_transition_matrix(self):
        """Plot the transition matrix as a heatmap."""
        transition_matrix = self.get_transition_matrix()
        
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        im = ax.imshow(transition_matrix, cmap='Blues')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Transition Probability')
        
        # Show all ticks and label them
        ax.set_xticks(np.arange(len(self.states)))
        ax.set_yticks(np.arange(len(self.states)))
        ax.set_xticklabels(self.states)
        ax.set_yticklabels(self.states)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                text = ax.text(j, i, f"{transition_matrix[i, j]:.2f}",
                               ha="center", va="center", color="white" if transition_matrix[i, j] > 0.5 else "black")
        
        ax.set_title("Transition Probability Matrix")
        plt.xlabel("To State")
        plt.ylabel("From State")
        plt.tight_layout()
        plt.show()
    
    def display_checkpoint_cards(self):
        """Display checkpoint cards in a grid."""
        checkpoint_results = self.simulator.checkpoint_results
        
        # Create HTML for checkpoint cards
        html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 20px;">'
        
        # Add cards for each checkpoint
        for i, checkpoint in enumerate(self.checkpoints):
            checkpoint_key = checkpoint
            if checkpoint_key in checkpoint_results:
                distribution = checkpoint_results[checkpoint_key]
                
                html += f'''
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px; background-color: white;">
                    <h3 style="margin-top: 0; text-align: center;">{self.checkpoint_labels[i]}</h3>
                    <div style="margin-top: 12px;">
                '''
                
                for state in self.states:
                    percentage = distribution.get(state, 0) * 100
                    state_color = colors.get(state, '#888')
                    html += f'''
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: {state_color};">{state}:</span>
                        <span style="font-weight: 500;">{percentage:.1f}%</span>
                    </div>
                    '''
                
                html += '</div></div>'
        
        # Add steady state card if available
        if "Steady-State" in checkpoint_results:
            steady_state = checkpoint_results["Steady-State"]
            
            html += f'''
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px; background-color: white;">
                <h3 style="margin-top: 0; text-align: center;">Steady State</h3>
                <div style="margin-top: 12px;">
            '''
            
            for state in self.states:
                percentage = steady_state.get(state, 0) * 100
                state_color = colors.get(state, '#888')
                html += f'''
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: {state_color};">{state}:</span>
                    <span style="font-weight: 500;">{percentage:.1f}%</span>
                </div>
                '''
            
            if hasattr(self.simulator, 'steady_state_time') and self.simulator.steady_state_time:
                html += f'''
                <div style="margin-top: 8px; text-align: center; font-size: 0.9em; color: #666;">
                    Reached after {self.format_time(self.simulator.steady_state_time)}
                </div>
                '''
            
            html += '</div></div>'
        
        html += '</div>'
        
        # Display the HTML
        display(HTML(html))
    
    def format_time(self, seconds):
        """Format time in seconds to a human-readable string."""
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
    
    def setup_dashboard(self):
        """Set up the interactive dashboard."""
        # Create tabs for different configuration sections
        tab1 = widgets.VBox([
            widgets.HTML('<h3>Simulation Parameters</h3>'),
            widgets.HBox([self.max_iterations_input, self.tolerance_input]),
            widgets.HBox([self.run_button])
        ])
        
        # Create transition matrix tab with normalization button
        matrix_widgets = []
        for from_state in self.states:
            state_header = widgets.HTML(f'<h4>From State: {from_state}</h4>')
            state_sliders = [self.transition_matrix_widgets[from_state][to_state] for to_state in self.states]
            matrix_widgets.extend([state_header] + state_sliders)
        
        tab2 = widgets.VBox([
            widgets.HTML('<h3>Transition Matrix</h3>'),
            self.normalize_button
        ] + matrix_widgets)
        
        # Create sojourn parameters tab
        sojourn_widgets = []
        for state in self.states:
            state_header = widgets.HTML(f'<h4>State: {state}</h4>')
            state_params = [
                self.sojourn_params_widgets[state]['shape'],
                self.sojourn_params_widgets[state]['scale']
            ]
            sojourn_widgets.extend([state_header] + state_params)
        
        tab3 = widgets.VBox([
            widgets.HTML('<h3>Sojourn Parameters</h3>')
        ] + sojourn_widgets)
        
        # Combine tabs
        tab = widgets.Tab(children=[tab1, tab2, tab3])
        tab.set_title(0, 'Simulation')
        tab.set_title(1, 'Transition Matrix')
        tab.set_title(2, 'Sojourn Parameters')
        
        # Display the dashboard
        display(widgets.VBox([
            widgets.HTML('<h2>Semi-Markov Process Simulation Dashboard</h2>'),
            tab,
            self.output
        ]))