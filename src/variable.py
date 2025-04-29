import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import u_V, u_F
import matplotlib.pyplot as plt
import numpy as np


class Variable(SubCircuit):
    """
    A variable subcircuit that attaches a capacitor from an input node to ground.
    This can be used to model a variable or state in a neural network.
    
    The input node (vi) will take a value between 0V and 1V, representing
    the state of the variable.
    
    Nodes:
    - vi: Input voltage node (variable state)
    - vdd: Supply voltage (typically 1V)
    - gnd: Ground
    
    The C value is passed as a constructor argument.
    """
    NODES = ('vi', 'vdd', 'gnd')
    
    def __init__(self, C, bounded=True):
        """
        Initialize the variable subcircuit with a specific capacitance value.
        
        Args:
            C (float): The capacitance value in Farads (default: 1pF)
            bounded (bool): If True, add diodes to bound vi between GND and Vdd
        """
        super().__init__(f'variable_{C}_{bounded}', *self.NODES)
        
        # Store capacitance value to make the subcircuit name unique
        self._C = C
        
        # Add a capacitor from vi to ground
        self.C(1, 'vi', 'gnd', C@u_F)
        
        # Add clamping diodes to bound the voltage if requested
        if bounded:
            # Create comparison node for upper clamp
            self.V('thresh_upper', 'thresh_upper', 'gnd', 0@u_V)
            # Create comparison node for lower clamp
            self.V('thresh_lower', 'thresh_lower', 'gnd', 0@u_V)
            
            # Lower bound (ground) using voltage-controlled switch
            # Switch turns on when vi < 0V (negative voltage), shorting vi to gnd
            self.S('lower_clamp', 'gnd', 'vi', 'thresh_lower', 'vi', model='VSWITCH')
            
            # Upper bound (vdd) using voltage-controlled switch
            # Switch turns on when vi > vdd (over voltage), shorting vi to vdd
            self.S('upper_clamp', 'vdd', 'vi', 'vi', 'vdd', model='VSWITCH')
            
            # Define the switch model for ideal diode behavior
            # VT=0: Switch at exactly 0V difference
            # VH=0mV: Keep at 0 or will make the simulator unstable
            # RON=0.1: Very low on resistance
            # ROFF=1e12: Very high off resistance
            self.model('VSWITCH', 'SW', VT=0, VH=0, RON=0.1, ROFF=1e12)


def main():
    """Run the Variable subcircuit simulation."""

    # TODO: This test is broken
    
    # Create a circuit
    circuit = Circuit('Variable Test')
    
    # Add power supply
    circuit.V('dd', 'vdd', circuit.gnd, 1@u_V)
    
    # Create variable subcircuits with different capacitance values
    C_values = [1e-12, 10e-12, 100e-12]  # 1pF, 10pF, 100pF
    variables = []
    
    for i, cap in enumerate(C_values):
        var = Variable(C=cap, bounded=True)
        variables.append(var)
        circuit.subcircuit(var)
        
        # Add a voltage source to charge the capacitor and configure it directly
        node_name = f'vi_{i}'
        # Create a pulse source directly instead of modifying it later
        circuit.PulseVoltageSource(f'i_{i}', node_name, circuit.gnd,
                                  initial_value=0@u_V, 
                                  pulsed_value=1.5@u_V, # Intentionally exceed Vdd to demonstrate clamping
                                  delay_time=0,
                                  rise_time=1e-9,
                                  fall_time=1e-9,
                                  pulse_width=1e-6,
                                  period=2e-6)
        
        # Instantiate the variable subcircuit
        circuit.X(f'var_{i}', var.name, node_name, 'vdd', circuit.gnd)
    
    # Create simulator
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    
    # Run transient analysis for 500ns
    analysis = simulator.transient(step_time=1e-9, end_time=500e-9)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.title('Variable Charging Behavior with Different C Values')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.grid(True)
    
    # Plot the charging curves for each C value
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(C_values)))
    
    for i, cap in enumerate(C_values):
        node_name = f'vi_{i}'
        plt.plot(analysis.time, analysis[node_name], 
                 color=colors[i], label=f'C = {cap*1e12:.0f} pF')
    
    # Calculate and plot theoretical charging curves
    for i, cap in enumerate(C_values):
        # For a simple RC circuit with R=0 (ideal voltage source), 
        # the capacitor would charge instantly
        # But in practice, there's always some resistance in the circuit
        # Let's assume a small resistance of 1kΩ for visualization
        r = 1e3  # 1kΩ
        tau = r * cap  # Time constant
        t = np.array(analysis.time)
        v = 1 - np.exp(-t / tau)  # Charging equation: V = Vfinal * (1 - e^(-t/τ))
        plt.plot(t, v, '--', color=colors[i], 
                 label=f'Theoretical (R=1kΩ, C={cap*1e12:.0f}pF)')
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot to a PNG file
    plt.savefig('graphs/variable_simulation_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'variable_simulation_results.png'")
    
    plt.show()
    
    # Print some information about the variable subcircuit
    print("\nVariable Subcircuit Information:")
    print("--------------------------------")
    print("The Variable subcircuit attaches a capacitor from vi to ground.")
    print("This can be used to model a variable or state in a neural network.")
    print("The voltage on the capacitor (between 0V and 1V) represents the state.")
    
    # Calculate and print the time constants for each C value
    print("\nTime Constants (assuming 1kΩ resistance):")
    for cap in C_values:
        tau = 1e3 * cap  # Time constant with 1kΩ resistance
        print(f"C = {cap*1e12:.0f} pF: τ = {tau*1e9:.2f} ns")


# Test the Variable subcircuit
if __name__ == '__main__':
    # Set up logging
    logger = Logging.setup_logging()
    main() 