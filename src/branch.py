import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import u_V
import matplotlib.pyplot as plt
import numpy as np

from nor import NorGate
from branch_voltage import BranchVoltage
from branch_current import BranchCurrent

from config import USE_AUX


class Branch(SubCircuit):
    """
    A complete branch subcircuit that chains a BranchVoltage subcircuit with a BranchCurrent subcircuit.
    
    The circuit takes two input voltages (Vi2 and Vi3) and processes them through the BranchVoltage
    subcircuit to produce an intermediate voltage (Vmi). This voltage is then fed into the BranchCurrent
    subcircuit to produce a current output (Imi).
    
    Nodes:
    - vam: Auxiliary voltage node
    - vi2: First input voltage
    - vi3: Second input voltage
    - imi_pos: Positive terminal for current output
    - imi_neg: Negative terminal for current output
    - vdd: Power supply
    - gnd: Ground
    
    The cmi2, cmi3, cmi values, and C parameter are passed as constructor arguments.
    """
    NODES = ('vam', 'vi2', 'vi3', 'imi_pos', 'imi_neg', 'vdd', 'gnd')
    
    def __init__(self, cmi, cmi2, cmi3, C):
        """
        Initialize the complete branch subcircuit with specific cmi2, cmi3, cmi values, and capacitance.
        
        Args:
            cmi (int): The ternary control value (-1, 0, or 1) for the BranchCurrent
            cmi2 (int): The ternary control value (-1, 0, or 1) for the first multiplexer in BranchVoltage
            cmi3 (int): The ternary control value (-1, 0, or 1) for the second multiplexer in BranchVoltage
            C (float): The capacitance value in Farads (default: 1pF)
        """
        super().__init__(f'branch_{cmi2}_{cmi3}_{cmi}_{C}', *self.NODES)
            
        # Store cmi values and C to make the subcircuit name unique
        self._cmi = cmi
        self._cmi2 = cmi2
        self._cmi3 = cmi3
        self._C = C
        
        # Create and add the BranchVoltage subcircuit
        branch_voltage = BranchVoltage(cmi2=cmi2, cmi3=cmi3)

        # Create and add the NOR gate for the auxiliary stage
        nor_gate = NorGate()
        
        # Create and add the BranchCurrent subcircuit with the C parameter
        branch_current = BranchCurrent(cmi=cmi, C=C)
        
        # Add the subcircuits
        self.subcircuit(branch_voltage)
        self.subcircuit(branch_current)
        self.subcircuit(nor_gate)
        
        # Create an internal node for connecting the voltage and current subcircuits
        v_stage12 = 'v_stage12'
        v_stage23 = 'v_stage23' if USE_AUX else 'v_stage12'
        
        # Instantiate the BranchVoltage subcircuit
        self.X('voltage_stage', branch_voltage.name, 'vi2', 'vi3', v_stage12, 'vdd', 'gnd')

        # Analog multiplier to multiply the output of the voltage stage with the auxiliary voltage
        if USE_AUX:
            self.B('multiplier', v_stage23, 'gnd', v='V(v_stage12)*V(vam)')
        
        # Instantiate the BranchCurrent subcircuit
        self.X('current_stage', branch_current.name, v_stage23, 'imi_pos', 'imi_neg', 'vdd', 'gnd')


# Helper function to calculate the theoretical output based on the mathematical model
def calculate_theoretical_output(vi2, vi3, cmi, cmi2, cmi3, C, vdd=1.0):
    """
    Calculate the theoretical output current based on the mathematical model:
    
    I_mi = c_mi * (C/4) * (mux2_output) * (mux3_output)
    
    Where mux output depends on cmi value:
    - cmi2/cmi3 = 1: VDD - Vi
    - cmi2/cmi3 = 0: VDD/2
    - cmi2/cmi3 = -1: Vi
    
    And the current conversion depends on cmi:
    - cmi = 1: I_mi = V_mi * (C/4)
    - cmi = 0: I_mi = 0
    - cmi = -1: I_mi = -V_mi * (C/4)
    
    Args:
        vi2: Input voltage 2
        vi3: Input voltage 3
        cmi2: Control value for multiplexer 2
        cmi3: Control value for multiplexer 3
        cmi: Control value for current converter
        C: Capacitance value in Farads
        vdd: Supply voltage (default 1.0V)
        
    Returns:
        Theoretical output current
    """
    # Calculate mux2 output
    if cmi2 == 1:
        mux2_out = vdd - vi2
    elif cmi2 == 0:
        mux2_out = vdd / 2
    else:  # cmi2 == -1
        mux2_out = vi2
    
    # Calculate mux3 output
    if cmi3 == 1:
        mux3_out = vdd - vi3
    elif cmi3 == 0:
        mux3_out = vdd / 2
    else:  # cmi3 == -1
        mux3_out = vi3
    
    # Calculate intermediate voltage (Vmi)
    vmi = mux2_out * mux3_out
    
    # Apply the current conversion with C/4 scaling factor
    return cmi * (C / 4) * vmi


# Test the complete branch subcircuit
def main():
    # Create a circuit
    circuit = Circuit('Complete Branch Test')
    
    # Add power supply
    vdd = 1
    circuit.V('dd', 'vdd', circuit.gnd, vdd@u_V)
    
    # Add input voltage sources
    v_i2 = circuit.V('i2', 'vi2', circuit.gnd, 0@u_V)  # Default input of 0V
    v_i3 = circuit.V('i3', 'vi3', circuit.gnd, 0@u_V)  # Default input of 0V
    circuit.V('am', 'vam', circuit.gnd, 1@u_V)  # Fix am to 1V for 0 influence
    
    # Test with different cmi combinations
    # We'll test a few representative combinations
    cmi_combinations = [
        (1, 1, 1),      # All positive correlation
        (0, 0, 0),      # All fixed/no current
        (-1, -1, -1),   # All negative correlation
        (1, -1, 1),     # Mixed correlations
        (-1, 1, -1),    # Mixed correlations
    ]
    
    # Set capacitance value
    C = 1e-9  # 1 nF
    
    # Create and test each branch
    results = []
    labels = []
    
    for idx, (cmi, cmi2, cmi3) in enumerate(cmi_combinations):
        # Create the complete branch subcircuit
        branch = Branch(cmi=cmi, cmi2=cmi2, cmi3=cmi3, C=C)
        circuit.subcircuit(branch)
        
        # Add measurement resistor (1 ohm) to convert current to voltage for measurement
        output_node = f'out{idx}'
        circuit.R(f'meas{idx}', output_node, circuit.gnd, 1)
        
        # Instantiate the branch with proper connections for current measurement
        circuit.X(f'branch{idx}', branch.name, 'vam', 'vi2', 'vi3', output_node, circuit.gnd, 'vdd', circuit.gnd)
        
        # Add to results tracking
        results.append(output_node)
        labels.append(f'cmi={cmi}, cmi2={cmi2}, cmi3={cmi3}')
    
    # Create simulator
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    
    # Test 1: Sweep vi2 while vi3 is low (0V)
    print("Simulating with vi3 = 0V")
    analysis1 = simulator.dc(Vi2=slice(0, vdd, 0.01))
    
    # Test 2: Sweep vi2 while vi3 is high (1V)
    print("Simulating with vi3 = {}V".format(vdd))
    v_i3.dc_value = vdd@u_V
    analysis2 = simulator.dc(Vi2=slice(0, vdd, 0.01))
    
    # Test 3: Sweep vi3 while vi2 is low (0V)
    print("Simulating with vi2 = 0V")
    v_i2.dc_value = 0@u_V
    analysis3 = simulator.dc(Vi3=slice(0, vdd, 0.01))
    
    # Test 4: Sweep vi3 while vi2 is high (1V)
    print("Simulating with vi2 = {}V".format(vdd))
    v_i2.dc_value = vdd@u_V
    analysis4 = simulator.dc(Vi3=slice(0, vdd, 0.01))
    
    # Create voltage sweep arrays for theoretical calculations
    v_sweep = np.linspace(0, vdd, 101)
    
    # Define a color cycle to ensure matching colors
    colors = plt.cm.tab10.colors
    
    # Plot the results
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot for sweeping vi2 with vi3=0
    ax = axes[0, 0]
    ax.set_title('Complete Branch Response - Sweeping vi2 (vi3=0V)')
    ax.set_xlabel('vi2 Voltage [V]')
    ax.set_ylabel('Output Current [A]')
    ax.grid(True)
    
    for idx, (cmi, cmi2, cmi3) in enumerate(cmi_combinations):
        color = colors[idx % len(colors)]
        # Plot simulated results with specific color
        sim_line, = ax.plot(analysis1['v-sweep'], analysis1[results[idx]], 
                           color=color, label=f'Simulated: cmi={cmi}, cmi2={cmi2}, cmi3={cmi3}')
        
        # Calculate and plot theoretical results with matching color
        theoretical = [calculate_theoretical_output(v, 0, cmi, cmi2, cmi3, C, vdd) for v in v_sweep]
        ax.plot(v_sweep, theoretical, '--', color=color, 
               label=f'Theoretical: cmi={cmi}, cmi2={cmi2}, cmi3={cmi3}')
    
    ax.legend()
    
    # Plot for sweeping vi2 with vi3=1
    ax = axes[0, 1]
    ax.set_title('Complete Branch Response - Sweeping vi2 (vi3=1V)')
    ax.set_xlabel('vi2 Voltage [V]')
    ax.set_ylabel('Output Current [A]')
    ax.grid(True)
    
    for idx, (cmi, cmi2, cmi3) in enumerate(cmi_combinations):
        color = colors[idx % len(colors)]
        # Plot simulated results with specific color
        ax.plot(analysis2['v-sweep'], analysis2[results[idx]], 
               color=color, label=f'Simulated: cmi={cmi}, cmi2={cmi2}, cmi3={cmi3}')
        
        # Calculate and plot theoretical results with matching color
        theoretical = [calculate_theoretical_output(v, vdd, cmi, cmi2, cmi3, C, vdd) for v in v_sweep]
        ax.plot(v_sweep, theoretical, '--', color=color, 
               label=f'Theoretical: cmi={cmi}, cmi2={cmi2}, cmi3={cmi3}')
    
    ax.legend()
    
    # Plot for sweeping vi3 with vi2=0
    ax = axes[1, 0]
    ax.set_title('Complete Branch Response - Sweeping vi3 (vi2=0V)')
    ax.set_xlabel('vi3 Voltage [V]')
    ax.set_ylabel('Output Current [A]')
    ax.grid(True)
    
    for idx, (cmi, cmi2, cmi3) in enumerate(cmi_combinations):
        color = colors[idx % len(colors)]
        # Plot simulated results with specific color
        ax.plot(analysis3['v-sweep'], analysis3[results[idx]], 
               color=color, label=f'Simulated: cmi={cmi}, cmi2={cmi2}, cmi3={cmi3}')
        
        # Calculate and plot theoretical results with matching color
        theoretical = [calculate_theoretical_output(0, v, cmi, cmi2, cmi3, C, vdd) for v in v_sweep]
        ax.plot(v_sweep, theoretical, '--', color=color, 
               label=f'Theoretical: cmi={cmi}, cmi2={cmi2}, cmi3={cmi3}')
    
    ax.legend()
    
    # Plot for sweeping vi3 with vi2=1
    ax = axes[1, 1]
    ax.set_title('Complete Branch Response - Sweeping vi3 (vi2=1V)')
    ax.set_xlabel('vi3 Voltage [V]')
    ax.set_ylabel('Output Current [A]')
    ax.grid(True)
    
    for idx, (cmi, cmi2, cmi3) in enumerate(cmi_combinations):
        color = colors[idx % len(colors)]
        # Plot simulated results with specific color
        ax.plot(analysis4['v-sweep'], analysis4[results[idx]], 
               color=color, label=f'Simulated: cmi={cmi}, cmi2={cmi2}, cmi3={cmi3}')
        
        # Calculate and plot theoretical results with matching color
        theoretical = [calculate_theoretical_output(vdd, v, cmi, cmi2, cmi3, C, vdd) for v in v_sweep]
        ax.plot(v_sweep, theoretical, '--', color=color, 
               label=f'Theoretical: cmi={cmi}, cmi2={cmi2}, cmi3={cmi3}')
    
    ax.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the formula
    
    # Save the plot to a PNG file
    plt.savefig('graphs/branch_simulation_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'branch_simulation_results.png'")
    
    plt.show()
    
    # Print truth table verification for a few key combinations
    print("\nComplete Branch Truth Table Verification:")
    
    # Test each cmi combination with specific input values
    test_voltages = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for cmi, cmi2, cmi3 in cmi_combinations:
        print(f"\ncmi={cmi}, cmi2={cmi2}, cmi3={cmi3}:")
        
        for vi2, vi3 in test_voltages:
            # Create a new circuit for each test point
            test_circuit = Circuit(f'Branch Test (cmi={cmi}, cmi2={cmi2}, cmi3={cmi3}, vi2={vi2}, vi3={vi3})')
            test_circuit.V('dd', 'vdd', test_circuit.gnd, vdd@u_V)
            test_circuit.V('i2', 'vi2', test_circuit.gnd, vi2@u_V)
            test_circuit.V('i3', 'vi3', test_circuit.gnd, vi3@u_V)
            test_circuit.V('am', 'vam', test_circuit.gnd, 1@u_V) # Fix am to 1V for 0 influence
            
            # Create and add the complete branch subcircuit
            branch = Branch(cmi=cmi, cmi2=cmi2, cmi3=cmi3, C=C)
            test_circuit.subcircuit(branch)
            
            # Add measurement resistor (1 ohm) to convert current to voltage for measurement
            test_circuit.R('meas', 'out', test_circuit.gnd, 1)
            
            # Instantiate the branch with proper connections for current measurement
            test_circuit.X('branch', branch.name, 'vam', 'vi2', 'vi3', 'out', circuit.gnd, 'vdd', circuit.gnd)
            
            # Run simulation
            test_simulator = test_circuit.simulator(temperature=25, nominal_temperature=25)
            analysis = test_simulator.operating_point()
            
            # Calculate theoretical output based on mathematical model
            theoretical_current = calculate_theoretical_output(vi2, vi3, cmi, cmi2, cmi3, C, vdd)
            
            # Extract output current (voltage across 1 ohm resistor equals current in amps)
            output_current = float(analysis['out'])
            
            print(f"vi2={vi2}V, vi3={vi3}V => Output Current = {output_current:.3e}A (Theoretical: {theoretical_current:.3e}A)")
            error = abs(output_current - theoretical_current)
            print(f"  Error: {error:.3e}A")
            assert error < 1e-12, "Output current does not match theoretical current"

if __name__ == '__main__':
    # Set up logging
    logger = Logging.setup_logging()
    main()