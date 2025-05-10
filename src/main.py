import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from pysat.formula import CNF

from clause import Clause
from variable import Variable


def load_cnf_file(cnf_file_path):
    """
    Load a CNF file using the pysat library.
    
    Args:
        cnf_file_path: Path to the CNF file
        
    Returns:
        A tuple containing (clauses, variable_names) where:
        - clauses is a list of clauses, each clause is a list of 3 tuples (var_idx, is_negated)
        - variable_names is a list of variable names
    """
    # Check if the file exists
    if not os.path.isfile(cnf_file_path):
        raise FileNotFoundError(f"CNF file not found: {cnf_file_path}")
    
    # Load the CNF formula using pysat
    cnf_formula = CNF(from_file=cnf_file_path)
    
    # Extract the number of variables
    num_vars = cnf_formula.nv
    
    # Create variable names (x1, x2, x3, ...)
    variable_names = [f'x{i}' for i in range(1, num_vars + 1)]
    
    # Convert pysat clauses to our format
    # pysat format: list of lists where each inner list contains integers
    # positive integers represent positive literals, negative integers represent negative literals
    # e.g., [1, -2, 3] means (x1 OR NOT x2 OR x3)
    clauses = []
    
    for pysat_clause in cnf_formula.clauses:
        # For 3-SAT, each clause should have exactly 3 literals
        # If a clause has fewer than 3 literals, we'll pad it with the first literal
        # If a clause has more than 3 literals, we'll only use the first 3
        
        # Ensure we have at least one literal
        if not pysat_clause:
            continue
            
        # Convert to our format: [(var_idx, is_negated), ...]
        clause = []
        for literal in pysat_clause[:3]:  # Take only the first 3 literals
            var_idx = abs(literal) - 1  # Convert from 1-indexed to 0-indexed
            is_negated = literal < 0
            clause.append((var_idx, is_negated))
        
        # Pad if necessary to ensure exactly 3 literals
        while len(clause) < 3:
            # Duplicate the first literal
            clause.append(clause[0])
            
        clauses.append(clause)
    
    return clauses, variable_names


def create_3sat_circuit(clauses, variable_names):
    """
    Create a circuit that represents a 3-SAT problem.
    
    Args:
        clauses: List of clauses, where each clause is a list of 3 tuples (var_idx, is_negated)
                 var_idx is the index in variable_names, is_negated is True if the variable is negated
        variable_names: List of variable names
        
    Returns:
        A PySpice Circuit object representing the 3-SAT problem
    """
    # Create the main circuit
    circuit = Circuit('3-SAT Solver')
    
    # Add power supply
    vdd = 1
    circuit.V('dd', 'vdd', circuit.gnd, vdd)
    
    # Create variable nodes
    var_nodes = [f'var_{name}' for name in variable_names]
    
    # Set a common capacitance value for all components
    C = 10e-9  # 10nF
    R_aux = 15e3  # 15kΩ
    C_aux = 10e-9  # 10nF
     
    # Create Variable subcircuits for each variable
    variables = []
    for i, name in enumerate(variable_names):
        var = Variable(C=C, bounded=True)
        variables.append(var)
        circuit.subcircuit(var)
        
        # Instantiate the Variable subcircuit
        circuit.X(f'var_{name}', var.name, var_nodes[i], 'vdd', circuit.gnd)
    
    # Create Clause subcircuits for each clause
    for i, clause in enumerate(clauses):
        # Convert the clause to cm values
        cm_values = []
        for var_idx, is_negated in clause:
            # cm = -1 if negated, 1 if not negated
            cm_values.append(-1 if is_negated else 1)
        
        # Create the Clause subcircuit
        clause_circuit = Clause(cm1=cm_values[0], cm2=cm_values[1], cm3=cm_values[2], C=C, R_aux=R_aux, C_aux=C_aux)
        circuit.subcircuit(clause_circuit)
        
        # Get the variable nodes for this clause
        v1_node = var_nodes[clause[0][0]]
        v2_node = var_nodes[clause[1][0]]
        v3_node = var_nodes[clause[2][0]]
        n1_node = f'n1_{i}'
        
        # Instantiate the Clause subcircuit
        circuit.X(f'clause_{i}', clause_circuit.name, n1_node, v1_node, v2_node, v3_node, 'vdd', circuit.gnd)
    
    return circuit


def run_3sat_simulation(circuit: Circuit, variable_names, clauses, simulation_time, step_time):
    """
    Run a transient simulation on the 3-SAT circuit.
    
    Args:
        circuit: The PySpice Circuit object
        variable_names: List of variable names
        simulation_time: Total simulation time in ms
        step_time: Time step for the simulation in ms
        
    Returns:
        The simulation results
    """
    # Clone the original circuit for transient analysis
    transient_circuit = circuit.clone(title='3-SAT Transient Analysis')
    
    # Run transient simulation with improved convergence parameters
    simulator = transient_circuit.simulator(temperature=25, nominal_temperature=25)
    
    # Add more options to improve convergence
    simulator.options(
        reltol=1e-2,     # Relaxed relative tolerance (was 1e-3)
        abstol=1e-5,     # Relaxed absolute tolerance (was 1e-6)
        itl1=500,        # Increased DC iteration limit (was 100)
        itl2=200,        # Increased DC transfer curve iteration limit (was 50)
        itl4=100,        # Transient analysis iteration limit
        gmin=1e-10,      # Minimum conductance
        method='gear',   # Integration method (alternatives: 'trap', 'euler')
        maxord=2         # Maximum order for integration method
    )

    # Print all available options
    # print(simulator._options)
    # raise Exception("Stop here")
    
    # Set initial conditions for variable nodes to help convergence
    # Initialize all variables to 0.5V (middle value)
    initial_conditions = {}
    for name in variable_names:
        node_name = f'var_{name}'
        initial_conditions[node_name] = 0.5  # Initialize to 0.5V

    for i in range(len(clauses)):
        node_name = f'n1_{i}'
        initial_conditions[node_name] = 1.0
    
    # Apply initial conditions
    if initial_conditions:
        simulator.initial_condition(**initial_conditions)
    
    # Run transient analysis
    analysis = simulator.transient(step_time=step_time * 1e-3, end_time=simulation_time * 1e-3)
    
    return analysis, simulation_time


MAX_CLAUSES_PLOTTED = 10

def plot_3sat_results_full(analysis, variable_names, clauses, file_name, show_plot=True):
    """
    Plot the results of the 3-SAT simulation with each variable on its own graph.
    
    Args:
        analysis: The simulation results
        variable_names: List of variable names
        file_name: Base name for the output file (without extension)
        show_plot: Whether to display the plot (default: True)
    """
    # Calculate the number of rows needed for the subplots
    num_variables = len(variable_names)
    num_clauses = min(len(clauses), MAX_CLAUSES_PLOTTED)
    
    # Create a figure with subplots - one for each variable and one for each clause
    fig, axes = plt.subplots(num_variables + num_clauses, 1, figsize=(12, 4 * (num_variables + num_clauses)), sharex=True)
    
    # If there's only one subplot, axes won't be an array
    if num_variables + num_clauses == 1:
        axes = [axes]
    
    # Plot each variable on its own subplot
    for i, name in enumerate(variable_names):
        node_name = f'var_{name}'
        ax = axes[i]
        
        # Check if the node exists in the analysis results
        if node_name in analysis.nodes:
            # Plot the variable voltage
            ax.plot(analysis.time * 1e3, analysis[node_name], label=f'Variable {name}', linewidth=2)
            
            # Add horizontal lines at 0.25V and 0.75V to indicate decision thresholds
            ax.axhline(y=0.25, color='r', linestyle='--', alpha=0.5, label='False threshold (0.25V)')
            ax.axhline(y=0.75, color='g', linestyle='--', alpha=0.5, label='True threshold (0.75V)')
            
            # Add a title and y-label for this subplot
            ax.set_title(f'Variable {name}')
            ax.set_ylabel('Voltage (V)')
            ax.grid(True)
            ax.legend()
            
            # Get the final voltage value
            final_voltage = float(analysis[node_name][-1])
            
            # Determine the final state
            if final_voltage > 0.75:
                state = "TRUE"
                color = 'green'
            elif final_voltage < 0.25:
                state = "FALSE"
                color = 'red'
            else:
                state = "UNDECIDED"
                color = 'orange'
                
            # Add text annotation showing the final state
            ax.text(0.98, 0.95, f'Final state: {state} ({final_voltage:.3f}V)', 
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes,
                    bbox=dict(facecolor=color, alpha=0.2))
        else:
            # If node not found, display an error message
            print(f"Warning: Node '{node_name}' not found in analysis results")
            ax.text(0.5, 0.5, f"Node '{node_name}' not found in analysis results",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    bbox=dict(facecolor='red', alpha=0.2))
            
            # List available nodes for debugging
            print(f"Available nodes: {', '.join(analysis.nodes)}")
    
    # fig2, axes2 = plt.subplots()

    # Plot each clause's vam node
    for i in range(num_clauses):
        ax = axes[num_variables + i] # if i != 1 else axes2
        node_name = f'xclause_{i}.vam'
        
        # Plot the clause vam voltage
        ax.plot(analysis.time * 1e3, analysis[node_name], label=f'Clause {i+1} vam', linewidth=2, color='purple')
        ax.plot(analysis.time * 1e3, analysis[f'n1_{i}'], label=f'Clause {i+1} n1', linewidth=2, color='green')
        
        # Add a title and y-label for this subplot
        ax.set_title(f'Clause {i+1} vam')
        ax.set_ylabel('Voltage (V)')
        ax.grid(True)
        ax.legend()
        
        # Get the final voltage value
        final_voltage = float(analysis[node_name][-1])
        
        # Add text annotation showing the final voltage
        ax.text(0.98, 0.95, f'Final voltage: {final_voltage:.3f}V', 
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                bbox=dict(facecolor='purple', alpha=0.2))
    
    # Add a common x-label for all subplots with proper formatting
    plt.xlabel('Time (milliseconds)')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'graphs/{file_name}.png', dpi=300, bbox_inches='tight')
    
    # Only show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def plot_3sat_results(analysis, variable_names, clauses, file_name, show_plot=True):
    """
    Plot the results of the 3-SAT simulation with variables overlayed, and clauses overlayed.
    
    Args:
        analysis: The simulation results
        variable_names: List of variable names
        file_name: Base name for the output file (without extension)
        show_plot: Whether to display the plot (default: True)
    """
    # Plot each variable on its own subplot
    for i, name in enumerate(variable_names):
        node_name = f'var_{name}'
        final_voltage = float(analysis[node_name][-1])
        satisfied_text = "True" if final_voltage > 0.5 else "False"
        plt.plot(analysis.time * 1e3, analysis[node_name], label=f'$V_{{{i + 1}}}$ ({satisfied_text})')
            
    # Add a title and y-label for this subplot
    # plt.title('Evolution of variable voltages over time')
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'graphs/{file_name}_variables.png', dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()

    # Plot each clause's vam node
    for i in range(len(clauses)):
        node_name = f'xclause_{i}.vam'
        plt.plot(analysis.time * 1e3, analysis[node_name], label=f'$V_{{a_{{{i + 1}}}}}$')
        
    # Add a title and y-label for this subplot
    # plt.title('Evolution of clause auxiliary node voltages over time')
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'graphs/{file_name}_clauses.png', dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_3sat_evolution(analysis, variable_names, clauses, file_name, show_plot=True):
    count_satisfied_list = []
    satisfied_at_idx = -1
    for i in range(len(analysis.time)):
        results, _ = interpret_results(analysis, variable_names, i)
        satisfies_all, count_satisfied, _ = verify_results(clauses, results, variable_names)
        count_satisfied_list.append(count_satisfied)
        if satisfies_all and satisfied_at_idx == -1:
            satisfied_at_idx = i

    times = np.array(analysis.time) * 1e3
    plt.plot(times, count_satisfied_list)
    # Also add a vertical line at time when satisfies_all is true
    plt.axvline(x=times[satisfied_at_idx], color='red', linestyle='--', label='All clauses satisfied')
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Number of clauses satisfied')
    plt.savefig(f'graphs/{file_name}_evolution.png', dpi=300, bbox_inches='tight')

    # Only show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

    return times[satisfied_at_idx]



def interpret_results(analysis, variable_names, at=-1):
    """
    Interpret the results of the 3-SAT simulation.
    
    Args:
        analysis: The simulation results
        variable_names: List of variable names
        
    Returns:
        A dictionary mapping variable names to boolean values
    """
    # Get the final voltage of each variable node
    final_voltages = {}
    for name in variable_names:
        node_name = f'var_{name}'
        final_voltage = float(analysis[node_name][at])
        final_voltages[name] = final_voltage
    
    # Interpret the voltages as boolean values
    # If voltage > 0.75V, the variable is True
    # If voltage < 0.25V, the variable is False
    # Otherwise, the variable is undecided
    results = {}
    for name, voltage in final_voltages.items():
        # if voltage > 0.75:
        #     results[name] = True
        # elif voltage < 0.25:
        #     results[name] = False
        # else:
        #     results[name] = None  # Undecided
        if voltage > 0.5:
            results[name] = True
        else:
            results[name] = False
    
    return results, final_voltages

def verify_results(clauses, results, variable_names):
    satisfies_all = True

    not_satisfied = []

    count_satisfied = 0
    for i, clause in enumerate(clauses):
        clause_satisfied = False
        for var_idx, is_negated in clause:
            var_name = variable_names[var_idx]
            var_value = results[var_name]
            if var_value is None:
                continue  # Skip undecided variables
            
            # Check if this literal satisfies the clause
            if (not is_negated and var_value) or (is_negated and not var_value):
                clause_satisfied = True
                break
        
        if not clause_satisfied:
            satisfies_all = False
            not_satisfied.append(i)
        else:
            count_satisfied += 1

    return satisfies_all, count_satisfied, not_satisfied

SIMULATION_TIME = 200
STEP_TIME = 10e-3

def main():
    """
    Main function to solve a 3-SAT problem from a CNF file.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Solve a 3-SAT problem using an analog circuit simulator.')
    parser.add_argument('--cnf', type=str, help='Path to the CNF file')
    parser.add_argument('--sim-time', type=float, default=SIMULATION_TIME, help='Simulation time in milliseconds')
    parser.add_argument('--step-time', type=float, default=STEP_TIME, help='Simulation step time in milliseconds')
    args = parser.parse_args()
    
    # If no CNF file is provided, use the default example
    if args.cnf:
        print(f"Loading CNF file: {args.cnf}")
        clauses, variable_names = load_cnf_file(args.cnf)
    else:
        raise ValueError("No CNF file provided")
    
    # Print the problem
    print("3-SAT Problem:")
    for i, clause in enumerate(clauses):
        clause_str = " or ".join([
            f"{'¬' if is_negated else ''}{variable_names[var_idx]}" 
            for var_idx, is_negated in clause
        ])
        print(f"Clause {i+1}: ({clause_str})")
    
    # Create the circuit
    circuit = create_3sat_circuit(clauses, variable_names)
    
    # Run the simulation
    print("\nRunning simulation...")
    analysis, _ = run_3sat_simulation(circuit, variable_names, clauses,
                                  simulation_time=args.sim_time, 
                                  step_time=args.step_time)
    
    # Plot the results
    plot_3sat_results(analysis, variable_names, clauses, args.cnf.split("/")[-1].split(".")[0])
    
    # Interpret the results
    results, final_voltages = interpret_results(analysis, variable_names)
    
    # Print the results
    print("\nSimulation Results:")
    print("------------------")
    for name, value in results.items():
        status = "True" if value else "False" if value is False else "Undecided"
        print(f"Variable {name}: {status} (Voltage: {final_voltages[name]:.3f}V)")
    
    # Check if the assignment satisfies all clauses
    satisfies_all, count_satisfied, not_satisfied = verify_results(clauses, results, variable_names)
    
    if satisfies_all:
        print(f"\nThe assignment satisfies all clauses! ({count_satisfied}/{len(clauses)})")
    else:
        print(f"\nThe assignment does not satisfy all clauses. ({count_satisfied}/{len(clauses)})")
    for clause in not_satisfied:
        print(f"Clause {clause + 1} is not satisfied!")

    plot_3sat_evolution(analysis, variable_names, clauses, args.cnf.split("/")[-1].split(".")[0])


if __name__ == '__main__':
    # Set up logging
    logger = Logging.setup_logging()
    main()
