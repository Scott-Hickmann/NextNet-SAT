import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit

from branch import Branch
from am_new import AMNewCircuit

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
    
    The C parameter is used for all Branch subcircuits.
    """
    NODES = ('n1', 'v1', 'v2', 'v3', 'vdd', 'gnd')
    
    def __init__(self, cm1, cm2, cm3, C, R_aux, C_aux):
        """
        Initialize the Clause subcircuit with specific cm1, cm2, cm3 values and capacitance.
        
        Args:
            cm1 (int): The first ternary control value (-1 or 1)
            cm2 (int): The second ternary control value (-1 or 1)
            cm3 (int): The third ternary control value (-1 or 1)
            C (float): The capacitance value in Farads (default: 1pF)
        """
        super().__init__(f'clause_{cm1}_{cm2}_{cm3}_{C}_{C_aux}', *self.NODES)
            
        # Store control values and capacitance
        self._cm1 = cm1
        self._cm2 = cm2
        self._cm3 = cm3
        self._C = C

        # 0 is not allowed
        if cm1 == 0 or cm2 == 0 or cm3 == 0:
            raise ValueError("0 is not allowed as a control value")
        
        am = AMNewCircuit(cm1=cm1, cm2=cm2, cm3=cm3, R=R_aux, C=C_aux)
        # am = AMFull(cm1=cm1, cm2=cm2, cm3=cm3, C=C_aux, gain=2)
        self.subcircuit(am)
        
        # Instantiate the AMFull subcircuit
        self.X('am', am.name, 'vam', 'n1', 'v1', 'v2', 'v3', 'vdd', 'gnd')
        
        # Create the three Branch subcircuits with different configurations
        branch1 = Branch(cmi=cm1, cmi2=cm2, cmi3=cm3, C=C)
        branch2 = Branch(cmi=cm2, cmi2=cm3, cmi3=cm1, C=C)
        branch3 = Branch(cmi=cm3, cmi2=cm1, cmi3=cm2, C=C)
        
        # Add the subcircuits
        self.subcircuit(branch1)
        self.subcircuit(branch2)
        self.subcircuit(branch3)
        
        # Instantiate the Branch subcircuits
        self.X('branch1', branch1.name, 'vam', 'v2', 'v3', 'v1', 'gnd', 'vdd', 'gnd')
        self.X('branch2', branch2.name, 'vam', 'v3', 'v1', 'v2', 'gnd', 'vdd', 'gnd')
        self.X('branch3', branch3.name, 'vam', 'v1', 'v2', 'v3', 'gnd', 'vdd', 'gnd')


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
    n1_node = 'n1'
    
    # Set control values
    cm1 = 1
    cm2 = 1
    cm3 = -1
    
    print(f"Testing Clause with cm1={cm1}, cm2={cm2}, cm3={cm3}")
    
    # Create the Clause subcircuit
    C = 1e-12  # 1nF
    R_aux = 10e3  # 10kΩ
    C_aux = 1e-6  # 10nF
    clause = Clause(cm1=cm1, cm2=cm2, cm3=cm3, C=C, R_aux=R_aux, C_aux=C_aux)
    circuit.subcircuit(clause)
    
    try:
        # Create a new circuit for transient analysis
        transient_circuit = Circuit('Clause Transient Analysis')
        
        # Add power supply
        transient_circuit.V('dd', 'vdd', transient_circuit.gnd, vdd)
        
        # Add capacitors from each node to ground
        # Using 1nF capacitors
        transient_circuit.C('c1', v1_node, transient_circuit.gnd, C)
        transient_circuit.C('c2', v2_node, transient_circuit.gnd, C)
        transient_circuit.C('c3', v3_node, transient_circuit.gnd, C)
        
        # Add the Clause subcircuit
        transient_circuit.subcircuit(clause)
        
        # Instantiate the Clause
        transient_circuit.X('clause', clause.name, n1_node, v1_node, v2_node, v3_node, 'vdd', transient_circuit.gnd)
        
        # Set initial conditions for the capacitors based on operating point
        # This helps avoid convergence issues at the start of the simulation
        v1_init = 0.5
        v2_init = 0.5
        v3_init = 0.5
        n1_init = 0.01
        # Run transient simulation
        transient_simulator = transient_circuit.simulator(temperature=25, nominal_temperature=25)
        transient_simulator.options(reltol=1e-3, abstol=1e-6, itl1=100, itl2=50)
        
        # Set initial conditions for the nodes
        transient_simulator.initial_condition(v1=v1_init, v2=v2_init, v3=v3_init, n1=n1_init)
        
        # Run transient analysis
        step_time = 1e-3 # 1 millisecond steps
        end_time = 60   # 60 seconds total
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
        i1_cap = C * dv1_dt
        i2_cap = C * dv2_dt
        i3_cap = C * dv3_dt
        
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