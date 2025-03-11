#!/usr/bin/env python3
"""
Run all circuit simulations in sequence.

This script imports and runs all the simulation functions from the various
circuit modules in the src directory.
"""

import os
import importlib
from time import time

# Ensure the graphs directory exists
os.makedirs('graphs', exist_ok=True)

# List of modules to run (in order)
modules = [
    'variable',
    'ternary_mux',
    'nor',
    'branch_current',
    'branch_voltage',
    'branch'
]

def main():
    start_time = time()
    
    print("=" * 80)
    print("RUNNING ALL CIRCUIT SIMULATIONS")
    print("=" * 80)
    
    # Run each module
    for i, module_name in enumerate(modules, 1):
        print(f"\n{'-' * 80}")
        print(f"[{i}/{len(modules)}] Running {module_name}...")
        print(f"{'-' * 80}\n")
        
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # If the module has a main function, call it
            if hasattr(module, 'main'):
                module.main()
            else:
                print(f"Warning: {module_name} does not have a main() function")
        
        except Exception as e:
            print(f"Error running {module_name}: {e}")
            print("Continuing with next module...")
    
    end_time = time()
    
    print("\n" + "=" * 80)
    print(f"ALL SIMULATIONS COMPLETED in {end_time - start_time:.2f} seconds")
    print("=" * 80)
    print("\nSimulation results saved to the 'graphs' directory:")
    
    # List all generated graph files
    graph_files = [f for f in os.listdir('graphs') if f.endswith('.png')]
    for graph_file in sorted(graph_files):
        print(f"- {graph_file}")

if __name__ == "__main__":
    main() 