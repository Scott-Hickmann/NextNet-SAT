import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import u_V
import matplotlib.pyplot as plt
import numpy as np

from ternary_mux import TernaryMultiplexer
from nor import NorGate


class Clause(SubCircuit):
    """
    A clause subcircuit that combines two ternary multiplexers and a NOR gate.
    
    The circuit takes two input voltages (Vi2 and Vi3) and processes them through
    ternary multiplexers conditioned on cmi2 and cmi3 respectively. The outputs
    of these multiplexers are then fed into a NOR gate to produce the final output Vmi.
    
    Nodes:
    - vi2: First input voltage
    - vi3: Second input voltage
    - vmi: Output voltage
    - vdd: Power supply
    - gnd: Ground
    
    The cmi2 and cmi3 values are passed as constructor arguments.
    """
    NODES = ('vi2', 'vi3', 'vmi', 'vdd', 'gnd')
    
    def __init__(self, cmi2=0, cmi3=0):
        """
        Initialize the clause subcircuit with specific cmi2 and cmi3 values.
        
        Args:
            cmi2 (int): The ternary control value (-1, 0, or 1) for the first multiplexer
            cmi3 (int): The ternary control value (-1, 0, or 1) for the second multiplexer
        """
        super().__init__(f'clause_{cmi2}_{cmi3}', *self.NODES)
            
        # Store cmi values to make the subcircuit name unique
        self._cmi2 = cmi2
        self._cmi3 = cmi3
        
        # Create and add the ternary multiplexers
        mux2 = TernaryMultiplexer(cmi=cmi2)
        mux3 = TernaryMultiplexer(cmi=cmi3)
        
        # Create and add the NOR gate
        nor_gate = NorGate()
        
        # Add the subcircuits
        self.subcircuit(mux2)
        self.subcircuit(mux3)
        self.subcircuit(nor_gate)
        
        # Instantiate the multiplexers
        self.X('mux2', mux2.name, 'vi2', 'mux2_out', 'vdd', 'gnd')
        self.X('mux3', mux3.name, 'vi3', 'mux3_out', 'vdd', 'gnd')
        
        # Instantiate the NOR gate
        self.X('nor1', nor_gate.NAME, 'mux2_out', 'mux3_out', 'vmi', 'vdd', 'gnd')


# Helper function to calculate the theoretical output based on the mathematical model
def calculate_theoretical_output(vi2, vi3, cmi2, cmi3, vdd=1.0):
    """
    Calculate the theoretical output based on the mathematical model:
    
    V_mi = (mux2_output) * (mux3_output)
    
    Where mux output depends on cmi value:
    - cmi = 1: VDD - Vi
    - cmi = 0: VDD/2
    - cmi = -1: Vi
    
    Args:
        vi2: Input voltage 2
        vi3: Input voltage 3
        cmi2: Control value for multiplexer 2
        cmi3: Control value for multiplexer 3
        vdd: Supply voltage (default 1.0V)
        
    Returns:
        Theoretical output voltage
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
    
    # Return the product
    return mux2_out * mux3_out


# Test the clause subcircuit
if __name__ == '__main__':
    # Set up logging
    logger = Logging.setup_logging()
    
    # Create a circuit
    circuit = Circuit('Clause Test')
    
    # Add power supply
    vdd = 1
    circuit.V('dd', 'vdd', circuit.gnd, vdd@u_V)
    
    # Add input voltage sources
    v_i2 = circuit.V('i2', 'vi2', circuit.gnd, 0@u_V)
    v_i3 = circuit.V('i3', 'vi3', circuit.gnd, 0@u_V)
    
    # Test with different cmi combinations
    # We'll test a few representative combinations
    cmi_combinations = [
        (1, 1),    # Both pass-through
        (0, 0),    # Both fixed at 0.5V
        (-1, -1),  # Both inverted
        (1, -1),   # First pass-through, second inverted
        (-1, 1),   # First inverted, second pass-through
        (1, 0),    # First pass-through, second fixed
        (0, 1),    # First fixed, second pass-through
    ]
    
    # Create and test each clause
    results = []
    labels = []
    
    for idx, (cmi2, cmi3) in enumerate(cmi_combinations):
        # Create the clause subcircuit
        clause = Clause(cmi2=cmi2, cmi3=cmi3)
        circuit.subcircuit(clause)
        
        # Instantiate the clause
        output_node = f'vmi{idx}'
        circuit.X(f'clause{idx}', clause.name, 'vi2', 'vi3', output_node, 'vdd', circuit.gnd)
        
        # Add to results tracking
        results.append(output_node)
        labels.append(f'cmi2={cmi2}, cmi3={cmi3}')
    
    # Create simulator
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    
    # Test 1: Sweep vi2 while vi3 is low
    print("Simulating with vi3 = 0V")
    analysis1 = simulator.dc(Vi2=slice(0, vdd, 0.01))
    
    # Test 2: Sweep vi2 while vi3 is high
    print("Simulating with vi3 = {}V".format(vdd))
    v_i3.dc_value = vdd@u_V
    analysis2 = simulator.dc(Vi2=slice(0, vdd, 0.01))
    
    # Test 3: Sweep vi3 while vi2 is low
    print("Simulating with vi2 = 0V")
    v_i2.dc_value = 0@u_V
    analysis3 = simulator.dc(Vi3=slice(0, vdd, 0.01))
    
    # Test 4: Sweep vi3 while vi2 is high
    print("Simulating with vi2 = {}V".format(vdd))
    v_i2.dc_value = vdd@u_V
    analysis4 = simulator.dc(Vi3=slice(0, vdd, 0.01))
    
    # Plot the results
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Create voltage sweep arrays for theoretical calculations
    v_sweep = np.linspace(0, vdd, 101)
    
    # Define a color cycle to ensure matching colors
    colors = plt.cm.tab10.colors
    
    # Plot for sweeping vi2 with vi3=0
    ax = axes[0, 0]
    ax.set_title('Clause Response - Sweeping vi2 (vi3=0V)')
    ax.set_xlabel('vi2 Voltage [V]')
    ax.set_ylabel('vmi Output Voltage [V]')
    ax.grid(True)
    
    for idx, (cmi2, cmi3) in enumerate(cmi_combinations):
        color = colors[idx % len(colors)]
        # Plot simulated results with specific color
        sim_line, = ax.plot(analysis1['v-sweep'], analysis1[results[idx]], 
                           color=color, label=f'Simulated: cmi2={cmi2}, cmi3={cmi3}')
        
        # Calculate and plot theoretical results with matching color
        theoretical = [calculate_theoretical_output(v, 0, cmi2, cmi3, vdd) for v in v_sweep]
        ax.plot(v_sweep, theoretical, '--', color=color, 
               label=f'Theoretical: cmi2={cmi2}, cmi3={cmi3}')
    
    ax.legend()
    
    # Plot for sweeping vi2 with vi3=1
    ax = axes[0, 1]
    ax.set_title('Clause Response - Sweeping vi2 (vi3=1V)')
    ax.set_xlabel('vi2 Voltage [V]')
    ax.set_ylabel('vmi Output Voltage [V]')
    ax.grid(True)
    
    for idx, (cmi2, cmi3) in enumerate(cmi_combinations):
        color = colors[idx % len(colors)]
        # Plot simulated results with specific color
        ax.plot(analysis2['v-sweep'], analysis2[results[idx]], 
               color=color, label=f'Simulated: cmi2={cmi2}, cmi3={cmi3}')
        
        # Calculate and plot theoretical results with matching color
        theoretical = [calculate_theoretical_output(v, vdd, cmi2, cmi3, vdd) for v in v_sweep]
        ax.plot(v_sweep, theoretical, '--', color=color, 
               label=f'Theoretical: cmi2={cmi2}, cmi3={cmi3}')
    
    ax.legend()
    
    # Plot for sweeping vi3 with vi2=0
    ax = axes[1, 0]
    ax.set_title('Clause Response - Sweeping vi3 (vi2=0V)')
    ax.set_xlabel('vi3 Voltage [V]')
    ax.set_ylabel('vmi Output Voltage [V]')
    ax.grid(True)
    
    for idx, (cmi2, cmi3) in enumerate(cmi_combinations):
        color = colors[idx % len(colors)]
        # Plot simulated results with specific color
        ax.plot(analysis3['v-sweep'], analysis3[results[idx]], 
               color=color, label=f'Simulated: cmi2={cmi2}, cmi3={cmi3}')
        
        # Calculate and plot theoretical results with matching color
        theoretical = [calculate_theoretical_output(0, v, cmi2, cmi3, vdd) for v in v_sweep]
        ax.plot(v_sweep, theoretical, '--', color=color, 
               label=f'Theoretical: cmi2={cmi2}, cmi3={cmi3}')
    
    ax.legend()
    
    # Plot for sweeping vi3 with vi2=1
    ax = axes[1, 1]
    ax.set_title('Clause Response - Sweeping vi3 (vi2=1V)')
    ax.set_xlabel('vi3 Voltage [V]')
    ax.set_ylabel('vmi Output Voltage [V]')
    ax.grid(True)
    
    for idx, (cmi2, cmi3) in enumerate(cmi_combinations):
        color = colors[idx % len(colors)]
        # Plot simulated results with specific color
        ax.plot(analysis4['v-sweep'], analysis4[results[idx]], 
               color=color, label=f'Simulated: cmi2={cmi2}, cmi3={cmi3}')
        
        # Calculate and plot theoretical results with matching color
        theoretical = [calculate_theoretical_output(vdd, v, cmi2, cmi3, vdd) for v in v_sweep]
        ax.plot(v_sweep, theoretical, '--', color=color, 
               label=f'Theoretical: cmi2={cmi2}, cmi3={cmi3}')
    
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print truth table verification for a few key combinations
    print("\nClause Truth Table Verification:")
    
    # Test each cmi combination with specific input values
    test_voltages = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for cmi2, cmi3 in cmi_combinations:
        print(f"\ncmi2={cmi2}, cmi3={cmi3}:")
        
        for vi2, vi3 in test_voltages:
            # Create a new circuit for each test point
            test_circuit = Circuit(f'Clause Test (cmi2={cmi2}, cmi3={cmi3}, vi2={vi2}, vi3={vi3})')
            test_circuit.V('dd', 'vdd', test_circuit.gnd, vdd@u_V)
            test_circuit.V('i2', 'vi2', test_circuit.gnd, vi2@u_V)
            test_circuit.V('i3', 'vi3', test_circuit.gnd, vi3@u_V)
            
            # Create and add the clause subcircuit
            clause = Clause(cmi2=cmi2, cmi3=cmi3)
            test_circuit.subcircuit(clause)
            test_circuit.X('clause', clause.name, 'vi2', 'vi3', 'vmi', 'vdd', test_circuit.gnd)
            
            # Run simulation
            test_simulator = test_circuit.simulator(temperature=25, nominal_temperature=25)
            analysis = test_simulator.operating_point()
            
            # Calculate expected multiplexer outputs based on cmi values
            mux2_out = vi2 if cmi2 == 1 else (0.5 if cmi2 == 0 else 1 - vi2)
            mux3_out = vi3 if cmi3 == 1 else (0.5 if cmi3 == 0 else 1 - vi3)
            
            # NOR gate output is 1 only when both inputs are 0
            # For analog values, we approximate: output ≈ 1 when both inputs are close to 0
            # and output ≈ 0 when either input is close to 1
            expected_nor = "high" if mux2_out < 0.2 and mux3_out < 0.2 else "low"
            
            # Calculate theoretical output based on mathematical model
            theoretical_output = calculate_theoretical_output(vi2, vi3, cmi2, cmi3, vdd)
            
            # Extract output value
            output_value = float(analysis['vmi'])
            output_state = "high" if output_value > 0.8 else ("mid" if output_value > 0.2 else "low")
            
            print(f"vi2={vi2}V, vi3={vi3}V => vmi={output_value:.3f}V (Expected NOR: {expected_nor}, Theoretical: {theoretical_output:.3f}V)") 