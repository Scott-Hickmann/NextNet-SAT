import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import u_V
import matplotlib.pyplot as plt
import numpy as np


class TernaryMultiplexer(SubCircuit):
    """
    A ternary multiplexer subcircuit that processes an input voltage based on a ternary control signal (cmi):
    - When cmi = 1: Output = Input voltage
    - When cmi = 0: Output = GND (0V)
    - When cmi = -1: Output = 1V - Input voltage
    
    Nodes:
    - input: The input voltage to be processed
    - output: The processed output voltage
    - vdd: Power supply
    - gnd: Ground
    
    The cmi value is passed as a constructor argument.
    """
    NODES = ('input', 'output', 'vdd', 'gnd')
    
    def __init__(self, cmi=0):
        """
        Initialize the ternary multiplexer with a specific cmi value.
        
        Args:
            cmi (int): The ternary control value (-1, 0, or 1)
        """
        super().__init__(f'ternary_mux_{cmi}', *self.NODES)
        
        # Validate cmi value
        if cmi not in [-1, 0, 1]:
            raise ValueError("cmi must be -1, 0, or 1")
            
        # Store cmi value to make the subcircuit name unique
        self._cmi = cmi
        
        # Define the NMOS and PMOS models with parameters
        self.model('NMOS', 'NMOS', vto=0, lambda_=1)
        self.model('PMOS', 'PMOS', vto=0, lambda_=1)
        
        # Set internal control signals based on cmi value
        cmi_pos_active = (cmi == 1)
        cmi_neg_active = (cmi == -1)
        
        # Create internal nodes for the control signals
        if cmi_pos_active:
            # Connect cmi_pos to VDD
            self.R('cmi_pos_pullup', 'cmi_pos', 'vdd', 1)  # Small resistance to VDD
        else:
            # Connect cmi_pos to GND
            self.R('cmi_pos_pulldown', 'cmi_pos', 'gnd', 1)  # Small resistance to GND
            
        if cmi_neg_active:
            # Connect cmi_neg to VDD
            self.R('cmi_neg_pullup', 'cmi_neg', 'vdd', 1)  # Small resistance to VDD
        else:
            # Connect cmi_neg to GND
            self.R('cmi_neg_pulldown', 'cmi_neg', 'gnd', 1)  # Small resistance to GND
        
        # Pass-through path (when cmi = 1, cmi_pos is high)
        # Transmission gate for input to output
        self.M('pass_p', 'input', 'cmi_neg', 'output', 'vdd', model='PMOS')  # PMOS turns on when cmi_neg is low (cmi = 1)
        self.M('pass_n', 'input', 'cmi_pos', 'output', 'gnd', model='NMOS')  # NMOS turns on when cmi_pos is high (cmi = 1)
        
        # Ground path (when cmi = 0, both cmi_pos and cmi_neg are low)
        # Pull-down to ground when neither cmi_pos nor cmi_neg is active
        self.M('gnd_n1', 'output', 'cmi_pos', 'node_gnd', 'gnd', model='NMOS')  # First NMOS in series
        self.M('gnd_n2', 'node_gnd', 'cmi_neg', 'gnd', 'gnd', model='NMOS')     # Second NMOS in series
        
        # Inverter path (when cmi = -1, cmi_neg is high)
        # Create 1V - input using a complementary circuit
        self.M('inv_p1', 'node_inv', 'input', 'vdd', 'vdd', model='PMOS')       # PMOS for inverter
        self.M('inv_n1', 'node_inv', 'input', 'gnd', 'gnd', model='NMOS')       # NMOS for inverter
        
        # Transmission gate for inverted signal to output
        self.M('inv_p2', 'node_inv', 'cmi_pos', 'output', 'vdd', model='PMOS')  # PMOS turns on when cmi_pos is low (cmi = -1)
        self.M('inv_n2', 'node_inv', 'cmi_neg', 'output', 'gnd', model='NMOS')  # NMOS turns on when cmi_neg is high (cmi = -1)


# Test the ternary multiplexer
if __name__ == '__main__':
    # Set up logging
    logger = Logging.setup_logging()
    
    # Create a circuit
    circuit = Circuit('Ternary Multiplexer Test')
    
    # Add power supply
    vdd = 1
    circuit.V('dd', 'vdd', circuit.gnd, vdd@u_V)
    
    # Add input voltage source
    v_in = circuit.V('in', 'in', circuit.gnd, 0.5@u_V)  # Default input of 0.5V
    
    # Test with different cmi values
    results = []
    labels = []
    
    # Create three different multiplexers with different cmi values
    mux1 = TernaryMultiplexer(cmi=1)
    mux0 = TernaryMultiplexer(cmi=0)
    mux_neg1 = TernaryMultiplexer(cmi=-1)
    
    # Add the subcircuits to the circuit
    circuit.subcircuit(mux1)
    circuit.subcircuit(mux0)
    circuit.subcircuit(mux_neg1)
    
    # Instantiate the multiplexers
    circuit.X('mux1', mux1.name, 'in', 'out1', 'vdd', circuit.gnd)
    circuit.X('mux0', mux0.name, 'in', 'out2', 'vdd', circuit.gnd)
    circuit.X('mux_neg1', mux_neg1.name, 'in', 'out3', 'vdd', circuit.gnd)
    
    # Create simulator
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    
    # Run DC sweep analysis
    analysis = simulator.dc(Vin=slice(0, vdd, 0.01))
    
    # Extract results
    results = [
        (analysis['v-sweep'], analysis['out1']),
        (analysis['v-sweep'], analysis['out2']),
        (analysis['v-sweep'], analysis['out3'])
    ]
    
    labels = [
        'cmi = 1 (pass-through)',
        'cmi = 0 (ground)',
        'cmi = -1 (1V - input)'
    ]
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.title('Ternary Multiplexer Response')
    plt.xlabel('Input Voltage [V]')
    plt.ylabel('Output Voltage [V]')
    plt.grid(True)
    
    for i, (x, y) in enumerate(results):
        plt.plot(x, y, label=labels[i])
    
    # Add ideal response lines for comparison
    x = np.linspace(0, vdd, 101)
    plt.plot(x, x, 'k--', alpha=0.5, label='Ideal: y = x')
    plt.plot(x, np.zeros_like(x), 'k--', alpha=0.5, label='Ideal: y = 0')
    plt.plot(x, vdd - x, 'k--', alpha=0.5, label='Ideal: y = 1 - x')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print some specific test points
    print("\nTernary Multiplexer Verification:")
    
    test_voltages = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Create a new circuit for point tests
    for cmi_value in [1, 0, -1]:
        # Create a new circuit for each cmi value
        test_circuit = Circuit(f'Ternary Multiplexer Test (cmi={cmi_value})')
        test_circuit.V('dd', 'vdd', test_circuit.gnd, vdd@u_V)
        
        # Create and add the multiplexer subcircuit
        mux = TernaryMultiplexer(cmi=cmi_value)
        test_circuit.subcircuit(mux)
        test_circuit.X('mux', mux.name, 'in', 'out', 'vdd', test_circuit.gnd)
        
        print(f"\ncmi = {cmi_value}:")
        for test_v in test_voltages:
            test_circuit.V('in', 'in', test_circuit.gnd, test_v@u_V)
            test_simulator = test_circuit.simulator(temperature=25, nominal_temperature=25)
            analysis = test_simulator.operating_point()
            print(f"Input = {test_v:.2f}V => Output ≈ {float(analysis['out']):.3f}V") 