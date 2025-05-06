import PySpice.Logging.Logging as Logging

from PySpice.Spice.Netlist import Circuit, SubCircuitFactory
from PySpice.Unit import u_V
import matplotlib.pyplot as plt
import numpy as np


class AndGate(SubCircuitFactory):
    NAME = 'and_gate'
    NODES = ('input_a', 'input_b', 'output', 'vdd', 'gnd')
    
    def __init__(self):
        super().__init__()
        
        # Define the NMOS and PMOS models with parameters
        self.model('NMOS', 'NMOS', vto=0, kp=1e-5, lambda_=200)
        self.model('PMOS', 'PMOS', vto=0, kp=1e-5, lambda_=200)
        
        # First implement NAND gate
        # PMOS transistors in parallel (from VDD to output)
        # M <name> <drain node> <gate node> <source node> <bulk/substrate node>
        self.M(1, 'nand_out', 'input_a', 'vdd', 'vdd', model='PMOS')
        self.M(2, 'nand_out', 'input_b', 'vdd', 'vdd', model='PMOS')
        
        # NMOS transistors in series (from output to GND)
        self.M(3, 'nand_out', 'input_a', 'node1', 'gnd', model='NMOS')
        self.M(4, 'node1', 'input_b', 'gnd', 'gnd', model='NMOS')
        
        # Inverter to convert NAND to AND
        self.M(5, 'output', 'nand_out', 'vdd', 'vdd', model='PMOS')
        self.M(6, 'output', 'nand_out', 'gnd', 'gnd', model='NMOS')


# Test plots for the AND gate
def main():
    # Create a circuit
    circuit = Circuit('AND Gate Test')
    
    # Add the AND gate subcircuit
    circuit.subcircuit(AndGate())
    
    # Instantiate the AND gate
    circuit.X('and1', 'and_gate', 'in_a', 'in_b', 'out', 'vdd', circuit.gnd)
    
    # Add power supply
    vdd = 1
    circuit.V('dd', 'vdd', circuit.gnd, vdd@u_V)
    
    # Add input voltage sources - store references to them
    v_in_a = circuit.V('in_a', 'in_a', circuit.gnd, 0@u_V)
    v_in_b = circuit.V('in_b', 'in_b', circuit.gnd, 0@u_V)
    
    # Create simulator
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    
    # Test 1: Sweep input A while input B is low
    print("Simulating with input B = 0V")
    analysis1 = simulator.dc(Vin_a=slice(0, vdd, 0.01))
    
    # Test 2: Sweep input A while input B is high
    print("Simulating with input B = {}V".format(vdd))
    v_in_b.dc_value = vdd@u_V
    analysis2 = simulator.dc(Vin_a=slice(0, vdd, 0.01))
    
    # Test 3: Sweep input B while input A is low
    print("Simulating with input A = 0V")
    v_in_a.dc_value = 0@u_V
    analysis3 = simulator.dc(Vin_b=slice(0, vdd, 0.01))
    
    # Test 4: Sweep input B while input A is high
    print("Simulating with input A = {}V".format(vdd))
    v_in_a.dc_value = vdd@u_V
    analysis4 = simulator.dc(Vin_b=slice(0, vdd, 0.01))
    
    # Test for analog multiplication behavior
    # For analog multiplication, we'll interpret the voltage inputs as normalized values
    # and the output as their product
    print("Simulating analog multiplication behavior")
    
    # Create arrays for normalized inputs and expected multiplication result
    num_points = 11
    input_values = np.linspace(0, 1, num_points)
    result_matrix = np.zeros((num_points, num_points))
    output_matrix = np.zeros((num_points, num_points))
    
    # Perform simulations for various input combinations
    for i, a_val in enumerate(input_values):
        for j, b_val in enumerate(input_values):
            # Set input voltages
            v_in_a.dc_value = (a_val * vdd)@u_V
            v_in_b.dc_value = (b_val * vdd)@u_V
            
            # Run operating point analysis
            analysis = simulator.operating_point()
            
            # Get output and store in matrix
            output_matrix[i, j] = float(analysis['out'][0])
            
            # Calculate expected multiplication result
            result_matrix[i, j] = a_val * b_val
    
    # Plot the results
    figure, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 8))
    
    # Plot for sweeping input A
    ax1.set_title('AND Gate Response - Sweeping Input A')
    ax1.set_xlabel('Input A Voltage [V]')
    ax1.set_ylabel('Output Voltage [V]')
    ax1.grid()
    ax1.plot(analysis1['v-sweep'], analysis1['out'], label='Input B = 0V')
    ax1.plot(analysis2['v-sweep'], analysis2['out'], label='Input B = {}V'.format(vdd))
    ax1.legend()
    
    # Plot for sweeping input B
    ax2.set_title('AND Gate Response - Sweeping Input B')
    ax2.set_xlabel('Input B Voltage [V]')
    ax2.set_ylabel('Output Voltage [V]')
    ax2.grid()
    ax2.plot(analysis3['v-sweep'], analysis3['out'], label='Input A = 0V')
    ax2.plot(analysis4['v-sweep'], analysis4['out'], label='Input A = {}V'.format(vdd))
    ax2.legend()
    
    # Plot the output matrix (actual AND gate output)
    im3 = ax3.imshow(output_matrix, extent=[0, vdd, 0, vdd], origin='lower', aspect='auto')
    ax3.set_title('Analog AND Gate Output')
    ax3.set_xlabel('Input B Voltage [V]')
    ax3.set_ylabel('Input A Voltage [V]')
    plt.colorbar(im3, ax=ax3, label='Output Voltage [V]')
    
    # Plot the expected multiplication result
    im4 = ax4.imshow(result_matrix, extent=[0, vdd, 0, vdd], origin='lower', aspect='auto')
    ax4.set_title('Expected Multiplication (A × B)')
    ax4.set_xlabel('Input B Voltage [V]')
    ax4.set_ylabel('Input A Voltage [V]')
    plt.colorbar(im4, ax=ax4, label='A × B')
    
    plt.tight_layout()

    # Save the plot to a PNG file
    plt.savefig('graphs/and_gate_simulation_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'and_gate_simulation_results.png'")
    
    plt.show()
    
    # Truth table verification
    print("\nAND Gate Truth Table Verification:")
    print(f"A=0, B=0 => Output ≈ {float(analysis1['out'][0]):.3f}V")
    print(f"A=1, B=0 => Output ≈ {float(analysis1['out'][-1]):.3f}V")
    print(f"A=0, B=1 => Output ≈ {float(analysis3['out'][-1]):.3f}V")
    print(f"A=1, B=1 => Output ≈ {float(analysis4['out'][-1]):.3f}V")
    
    # Verify multiplication behavior
    print("\nAnalog Multiplication Verification (Selected Points):")
    a_vals = [0.0, 0.2, 0.5, 0.8, 1.0]
    b_vals = [0.0, 0.2, 0.5, 0.8, 1.0]
    
    print("      | " + " | ".join([f"B={b:.1f}" for b in b_vals]))
    print("------" + "+-------" * len(b_vals))
    
    for a in a_vals:
        i = int(a * (num_points - 1))
        row = [f"A={a:.1f}"]
        for b in b_vals:
            j = int(b * (num_points - 1))
            row.append(f"{output_matrix[i, j]:.3f}")
        print(" | ".join(row))

if __name__ == '__main__':
    # Set up logging
    logger = Logging.setup_logging()
    main() 