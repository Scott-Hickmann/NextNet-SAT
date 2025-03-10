import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import u_V
import matplotlib.pyplot as plt
import numpy as np


class TernaryMultiplexer(SubCircuit):
    """
    A ternary multiplexer subcircuit that processes an input voltage based on a ternary control signal (cmi):
    - When cmi = 1: Output = Input voltage
    - When cmi = 0: Output = 0.5V (fixed)
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
        # Use more ideal parameters for better switching behavior
        self.model('NMOS', 'NMOS', vto=0.2, kp=200, lambda_=0.01)
        self.model('PMOS', 'PMOS', vto=-0.2, kp=100, lambda_=0.01)
        
        # Create internal control signals
        # For cmi = 1: cmi_pos = VDD, cmi_neg = GND
        # For cmi = 0: cmi_pos = GND, cmi_neg = GND
        # For cmi = -1: cmi_pos = GND, cmi_neg = VDD
        
        # Create fixed voltage sources for control signals instead of resistors
        if cmi == 1:
            self.V('cmi_pos', 'cmi_pos', 'gnd', 1@u_V)  # cmi_pos = VDD
            self.V('cmi_neg', 'cmi_neg', 'gnd', 0@u_V)  # cmi_neg = GND
        elif cmi == -1:
            self.V('cmi_pos', 'cmi_pos', 'gnd', 0@u_V)  # cmi_pos = GND
            self.V('cmi_neg', 'cmi_neg', 'gnd', 1@u_V)  # cmi_neg = VDD
        else:  # cmi == 0
            self.V('cmi_pos', 'cmi_pos', 'gnd', 0@u_V)  # cmi_pos = GND
            self.V('cmi_neg', 'cmi_neg', 'gnd', 0@u_V)  # cmi_neg = GND
        
        # CASE 1: Pass-through path (when cmi = 1)
        if cmi == 1:
            # Simple direct connection for pass-through
            self.R('pass_r', 'input', 'output', 0.1)  # Low resistance connection
        
        # CASE 2: Fixed 0.5V output (when cmi = 0)
        elif cmi == 0:
            # Direct connection to a 0.5V source
            self.V('half_vdd', 'output', 'gnd', 0.5@u_V)  # Fixed 0.5V output
        
        # CASE 3: Inverter path (when cmi = -1)
        elif cmi == -1:
            # Create a voltage-controlled voltage source for 1V - input
            # E <name> <out+> <out-> <in+> <in-> <gain>
            self.VCVS('inv', 'output', 'gnd', 'vdd', 'input', 1)


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
        'cmi = 0 (fixed 0.5V)',
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
    plt.plot(x, np.full_like(x, 0.5), 'k--', alpha=0.5, label='Ideal: y = 0.5')
    plt.plot(x, vdd - x, 'k--', alpha=0.5, label='Ideal: y = 1 - x')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Calculate and print the mean squared error for each case
    print("\nMean Squared Error (MSE) from ideal response:")
    
    # Calculate MSE for cmi = 1 (pass-through)
    mse1 = np.mean((analysis['out1'] - analysis['v-sweep'])**2)
    print(f"cmi = 1 (pass-through): {mse1:.6f}")
    
    # Calculate MSE for cmi = 0 (fixed 0.5V)
    mse0 = np.mean((analysis['out2'] - 0.5)**2)
    print(f"cmi = 0 (fixed 0.5V): {mse0:.6f}")
    
    # Calculate MSE for cmi = -1 (1V - input)
    mse_neg1 = np.mean((analysis['out3'] - (vdd - analysis['v-sweep']))**2)
    print(f"cmi = -1 (1V - input): {mse_neg1:.6f}")
    
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
            ideal = test_v if cmi_value == 1 else (0.5 if cmi_value == 0 else vdd - test_v)
            print(f"Input = {test_v:.2f}V => Output = {float(analysis['out']):.3f}V (Ideal: {ideal:.3f}V, Error: {abs(float(analysis['out']) - ideal):.3f}V)") 