import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit

from branch import Branch


class Clause(SubCircuit):
    """
    A Clause subcircuit that contains three Branch subcircuits with specific configurations.
    
    The circuit takes three input/output nodes (v1, v2, v3) and connects them to three Branch
    subcircuits with specific configurations of control values (cm1, cm2, cm3).
    
    Each Branch subcircuit's current output flows into one of the nodes:
    - Branch 1 (vi2=v2, vi3=v3, cmi=cm1, cmi2=cm2, cmi3=cm3) flows into v1
    - Branch 2 (vi2=v3, vi3=v1, cmi=cm2, cmi2=cm3, cmi3=cm1) flows into v2
    - Branch 3 (vi2=v1, vi3=v2, cmi=cm3, cmi2=cm1, cmi3=cm2) flows into v3
    
    Nodes:
    - v1: First voltage node (input/output)
    - v2: Second voltage node (input/output)
    - v3: Third voltage node (input/output)
    - vdd: Power supply
    - gnd: Ground
    """
    NODES = ('v1', 'v2', 'v3', 'vdd', 'gnd')
    
    def __init__(self, cm1=0, cm2=0, cm3=0):
        """
        Initialize the Clause subcircuit with specific cm1, cm2, cm3 values.
        
        Args:
            cm1 (int): The first ternary control value (-1, 0, or 1)
            cm2 (int): The second ternary control value (-1, 0, or 1)
            cm3 (int): The third ternary control value (-1, 0, or 1)
        """
        super().__init__(f'clause_{cm1}_{cm2}_{cm3}', *self.NODES)
            
        # Store control values
        self._cm1 = cm1
        self._cm2 = cm2
        self._cm3 = cm3
        
        # Create the three Branch subcircuits with different configurations
        branch1 = Branch(cmi=cm1, cmi2=cm2, cmi3=cm3)
        branch2 = Branch(cmi=cm2, cmi2=cm3, cmi3=cm1)
        branch3 = Branch(cmi=cm3, cmi2=cm1, cmi3=cm2)
        
        # Add the subcircuits
        self.subcircuit(branch1)
        self.subcircuit(branch2)
        self.subcircuit(branch3)
        
        # Instantiate Branch 1: vi2=v2, vi3=v3, current flows into v1
        self.X('branch1', branch1.name, 'v2', 'v3', 'v1', 'gnd', 'vdd', 'gnd')
        
        # Instantiate Branch 2: vi2=v3, vi3=v1, current flows into v2
        self.X('branch2', branch2.name, 'v3', 'v1', 'v2', 'gnd', 'vdd', 'gnd')
        
        # Instantiate Branch 3: vi2=v1, vi3=v2, current flows into v3
        self.X('branch3', branch3.name, 'v1', 'v2', 'v3', 'gnd', 'vdd', 'gnd')


def main():
    """
    Test function for the Clause subcircuit using capacitors for current measurement.
    Includes time analysis and plotting of currents over time.
    """
    # Import plotting libraries
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a circuit
    circuit = Circuit('Clause Test with Capacitors')
    
    # Add power supply
    vdd = 1
    circuit.V('dd', 'vdd', circuit.gnd, vdd)
    
    # Create nodes for the clause
    v1_node = 'v1'
    v2_node = 'v2'
    v3_node = 'v3'
    
    # Add initial voltage sources to set node voltages
    # This helps establish an initial operating point
    circuit.V('v1_init', v1_node, circuit.gnd, 0.5)
    circuit.V('v2_init', v2_node, circuit.gnd, 0.5)
    circuit.V('v3_init', v3_node, circuit.gnd, 0.5)
    
    # Set control values
    cm1 = 0
    cm2 = 1
    cm3 = 0
    
    print(f"Testing Clause with cm1={cm1}, cm2={cm2}, cm3={cm3}")
    
    # Create the Clause subcircuit
    clause = Clause(cm1=cm1, cm2=cm2, cm3=cm3)
    circuit.subcircuit(clause)
    
    # Instantiate the Clause
    circuit.X('clause', clause.name, v1_node, v2_node, v3_node, 'vdd', circuit.gnd)
    
    # Run simulation to establish operating point
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    simulator.options(reltol=1e-3, abstol=1e-6, itl1=100, itl2=50)
    
    try:
        # Run operating point analysis
        op_analysis = simulator.operating_point()
        print("Operating point analysis successful")
        
        # Create a new circuit for transient analysis
        transient_circuit = Circuit('Clause Transient Analysis')
        
        # Add power supply
        transient_circuit.V('dd', 'vdd', transient_circuit.gnd, vdd)
        
        # Add capacitors from each node to ground
        # Using 1nF capacitors
        capacitance = 1e-9  # 1nF
        transient_circuit.C('c1', v1_node, transient_circuit.gnd, capacitance)
        transient_circuit.C('c2', v2_node, transient_circuit.gnd, capacitance)
        transient_circuit.C('c3', v3_node, transient_circuit.gnd, capacitance)
        
        # Add small resistors in parallel with capacitors to improve convergence
        # These high-value resistors (1MΩ) won't significantly affect the circuit behavior
        transient_circuit.R('r1', v1_node, transient_circuit.gnd, 1e6)
        transient_circuit.R('r2', v2_node, transient_circuit.gnd, 1e6)
        transient_circuit.R('r3', v3_node, transient_circuit.gnd, 1e6)
        
        # Add the Clause subcircuit
        transient_circuit.subcircuit(clause)
        
        # Instantiate the Clause
        transient_circuit.X('clause', clause.name, v1_node, v2_node, v3_node, 'vdd', transient_circuit.gnd)
        
        # Set initial conditions for the capacitors based on operating point
        # This helps avoid convergence issues at the start of the simulation
        v1_init = float(op_analysis[v1_node][0])
        v2_init = float(op_analysis[v2_node][0])
        v3_init = float(op_analysis[v3_node][0])
        
        # Run transient simulation
        transient_simulator = transient_circuit.simulator(temperature=25, nominal_temperature=25)
        transient_simulator.options(reltol=1e-3, abstol=1e-6, itl1=100, itl2=50)
        
        # Set initial conditions for the nodes
        transient_simulator.initial_condition(v1=v1_init, v2=v2_init, v3=v3_init)
        
        # Run transient analysis
        step_time = 1e-6  # 1 microsecond steps
        end_time = 1e-3   # 1 millisecond total
        analysis = transient_simulator.transient(step_time=step_time, end_time=end_time)
        
        # Convert PySpice values to plain NumPy arrays to avoid Unit conversion issues
        # Extract time points as plain floating-point values
        time_array = np.array([float(t) for t in analysis.time])
        
        # Extract voltage waveforms as plain floating-point values
        v1_array = np.array([float(v) for v in analysis[v1_node]])
        v2_array = np.array([float(v) for v in analysis[v2_node]])
        v3_array = np.array([float(v) for v in analysis[v3_node]])
        
        # Calculate currents through the capacitors: I = C * dV/dt
        # We'll use numpy's gradient function to calculate dV/dt
        dv1_dt = np.gradient(v1_array, time_array)
        dv2_dt = np.gradient(v2_array, time_array)
        dv3_dt = np.gradient(v3_array, time_array)
        
        # Calculate capacitor currents
        i1_cap = capacitance * dv1_dt
        i2_cap = capacitance * dv2_dt
        i3_cap = capacitance * dv3_dt
        
        # Calculate resistor currents: I = V/R
        i1_res = v1_array / 1e6
        i2_res = v2_array / 1e6
        i3_res = v3_array / 1e6
        
        # Calculate total currents (capacitor + resistor)
        i1 = i1_cap + i1_res
        i2 = i2_cap + i2_res
        i3 = i3_cap + i3_res
        
        # Print final values
        print("\nFinal node voltages:")
        print(f"v1: {v1_array[-1]:.6f}V")
        print(f"v2: {v2_array[-1]:.6f}V")
        print(f"v3: {v3_array[-1]:.6f}V")
        
        # Plot the results
        plt.figure(figsize=(12, 10))
        
        # Plot voltages
        plt.subplot(3, 1, 1)
        plt.plot(time_array * 1e6, v1_array, 'r-', label='v1')
        plt.plot(time_array * 1e6, v2_array, 'g-', label='v2')
        plt.plot(time_array * 1e6, v3_array, 'b-', label='v3')
        plt.xlabel('Time (μs)')
        plt.ylabel('Voltage (V)')
        plt.title(f'Node Voltages Over Time (cm1={cm1}, cm2={cm2}, cm3={cm3})')
        plt.grid(True)
        plt.legend()
        
        # Plot capacitor currents
        plt.subplot(3, 1, 2)
        plt.plot(time_array * 1e6, i1_cap * 1e9, 'r-', label='i1_cap')
        plt.plot(time_array * 1e6, i2_cap * 1e9, 'g-', label='i2_cap')
        plt.plot(time_array * 1e6, i3_cap * 1e9, 'b-', label='i3_cap')
        plt.xlabel('Time (μs)')
        plt.ylabel('Capacitor Current (nA)')
        plt.title('Capacitor Currents Over Time')
        plt.grid(True)
        plt.legend()
        
        # Plot total currents
        plt.subplot(3, 1, 3)
        plt.plot(time_array * 1e6, i1 * 1e9, 'r-', label='i1_total')
        plt.plot(time_array * 1e6, i2 * 1e9, 'g-', label='i2_total')
        plt.plot(time_array * 1e6, i3 * 1e9, 'b-', label='i3_total')
        plt.xlabel('Time (μs)')
        plt.ylabel('Total Current (nA)')
        plt.title('Total Node Currents Over Time')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()

        # Save the plot to a PNG file
        plt.savefig('graphs/clause_simulation_results.png', dpi=300, bbox_inches='tight')
        print("Plot saved to 'clause_simulation_results.png'")
        
        plt.show()
        
        # Print peak currents
        print("\nPeak capacitor currents:")
        print(f"Peak current into v1: {np.max(np.abs(i1_cap)) * 1e9:.6f} nA")
        print(f"Peak current into v2: {np.max(np.abs(i2_cap)) * 1e9:.6f} nA")
        print(f"Peak current into v3: {np.max(np.abs(i3_cap)) * 1e9:.6f} nA")
        
        print("\nPeak total currents:")
        print(f"Peak total current into v1: {np.max(np.abs(i1)) * 1e9:.6f} nA")
        print(f"Peak total current into v2: {np.max(np.abs(i2)) * 1e9:.6f} nA")
        print(f"Peak total current into v3: {np.max(np.abs(i3)) * 1e9:.6f} nA")
        
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Set up logging
    logger = Logging.setup_logging()
    main() 