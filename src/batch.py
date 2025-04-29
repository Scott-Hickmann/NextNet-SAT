#!/usr/bin/env python3
import os
import csv
import glob
import json
from datetime import datetime
import re

from main import load_cnf_file, create_3sat_circuit, run_3sat_simulation, interpret_results, verify_results, plot_3sat_results, plot_3sat_evolution

# Configuration
CNF_FOLDER = 'cnf'
RESULTS_CSV = 'results/results.csv'
GRAPHS_FOLDER = 'graphs'
SIMULATION_TIME = 20  # seconds
STEP_TIME = 1e-3  # seconds

def run_simulation(cnf_file):
    """
    Run the 3-SAT simulation on a single CNF file and return the results.
    
    Args:
        cnf_file: Path to the CNF file
        
    Returns:
        A dictionary with the simulation results
    """
    print(f"Processing {cnf_file}...")
    
    # Load the CNF file
    clauses, variable_names = load_cnf_file(cnf_file)
    
    # Create the circuit
    circuit = create_3sat_circuit(clauses, variable_names)
    
    # Run the simulation
    analysis, simulation_time = run_3sat_simulation(circuit, variable_names, clauses,
                                  simulation_time=SIMULATION_TIME, 
                                  step_time=STEP_TIME)
    
    # Generate a plot filename based on the CNF filename
    base_filename = os.path.splitext(os.path.basename(cnf_file))[0]
    
    # Plot the evolution and save to file (without showing the plot)
    plot_3sat_results(analysis, variable_names, clauses, base_filename, show_plot=False)
    satisfied_at = plot_3sat_evolution(analysis, variable_names, clauses, base_filename, show_plot=False)
    
    # Interpret the results
    results, final_voltages = interpret_results(analysis, variable_names)
    
    # Count satisfied clauses
    all_satisfied, satisfied_count, not_satisfied = verify_results(clauses, results, variable_names)
    print(f"Not satisfied clauses (1-indexed): {[x + 1 for x in not_satisfied]}")
    
    # Prepare the result dictionary
    result = {
        'file_name': os.path.basename(cnf_file),
        'satisfied_clauses': satisfied_count,
        'total_clauses': len(clauses),
        'all_satisfied': all_satisfied,
        'satisfied_at': f"{satisfied_at:.3f}",
        'finished_at': f"{simulation_time:.3f}",
        'voltages': final_voltages,
        'plot_file': f'graphs/{base_filename}.png',
    }
    
    return result

def custom_sort_key(filename):
    base_name = os.path.basename(filename)
    match = re.match(r"(.+)-(\d+)\.cnf$", base_name)
    
    if match:
        name, number = match.groups()
        return (name, int(number))  # Sort by name, then numerically by number
    return (base_name, float('inf'))  # Sort alphabetically, place non-numbered files

def main():
    """
    Main function to run the batch processing.
    """
    # Create the results and graphs directories if they don't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs(GRAPHS_FOLDER, exist_ok=True)
    
    # Create or open the CSV file
    csv_exists = os.path.isfile(RESULTS_CSV)
    
    with open(RESULTS_CSV, 'a', newline='') as csvfile:
        # Define the CSV columns
        fieldnames = ['timestamp', 'file_name', 'satisfied_clauses', 'total_clauses', 
                     'all_satisfied', 'satisfied_at', 'finished_at', 'voltages', 'plot_file']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header if the file is new
        if not csv_exists:
            writer.writeheader()

        # Get all CNF files in the CNF folder
        cnf_files = glob.glob(os.path.join(CNF_FOLDER, '*.cnf'))
        cnf_files.sort(key=custom_sort_key)
        # Filter out files already in the CSV
        processed_files = set()
        
        # Check if the CSV file exists and has content
        if csv_exists and os.path.getsize(RESULTS_CSV) > 0:
            # Read the CSV to get already processed files
            with open(RESULTS_CSV, 'r', newline='') as existing_csv:
                reader = csv.DictReader(existing_csv)
                for row in reader:
                    processed_files.add(row['file_name'])
        
        # Filter out already processed files
        cnf_files = [f for f in cnf_files if os.path.basename(f) not in processed_files]
        print(cnf_files)
        
        if not cnf_files:
            print(f"No new CNF files found in {CNF_FOLDER} folder.")
            return
        
        # Process each CNF file
        for cnf_file in cnf_files:
            try:
                # Run the simulation
                result = run_simulation(cnf_file)
                
                # Add timestamp
                result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Convert voltages to JSON string for CSV storage
                result['voltages'] = json.dumps(result['voltages'])
                
                # Write the result to the CSV
                writer.writerow(result)
                
                # Flush to ensure data is written immediately
                csvfile.flush()
                
                print(f"Results for {cnf_file}:")
                print(f"  Satisfied clauses: {result['satisfied_clauses']}/{result['total_clauses']}")
                print(f"  All satisfied: {result['all_satisfied']}")
                print(f"  Plot saved to: {result['plot_file']}")
                print()
                
            except Exception as e:
                print(f"Error processing {cnf_file}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print(f"Batch processing complete. Results saved to {RESULTS_CSV}")

if __name__ == '__main__':
    main() 