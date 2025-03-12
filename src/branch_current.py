import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import u_V
import matplotlib.pyplot as plt
import numpy as np


class BranchCurrent(SubCircuit):
    """
    A voltage-to-current converter subcircuit that converts an input voltage Vmi
    into a current Imi where Imi = cmi*Vmi*(C/4).
    
    The circuit takes an input voltage (Vmi) and produces a current (Imi) that is
    proportional to the input voltage, with the proportionality constant determined
    by the cmi value and scaled by C/4.
    
    Nodes:
    - vmi: Input voltage
    - imi_pos: Positive terminal for current output
    - imi_neg: Negative terminal for current output
    - vdd: Power supply
    - gnd: Ground
    
    The cmi value is passed as a constructor argument and must be -1, 0, or 1:
    - When cmi = 1: Imi = Vmi*(C/4) (positive correlation)
    - When cmi = 0: Imi = 0 (no current)
    - When cmi = -1: Imi = -Vmi*(C/4) (negative correlation)
    """
    NODES = ('vmi', 'imi_pos', 'imi_neg', 'vdd', 'gnd')
    
    def __init__(self, cmi, C):
        """
        Initialize the voltage-to-current converter with a specific cmi value and capacitance.
        
        Args:
            cmi (int): The ternary control value (-1, 0, or 1)
            C (float): The capacitance value in Farads (default: 1pF)
        """
        super().__init__(f'branch_current_{cmi}_{C}', *self.NODES)
        
        # Validate cmi value
        if cmi not in [-1, 0, 1]:
            raise ValueError("cmi must be -1, 0, or 1")
            
        # Store cmi and C values to make the subcircuit name unique
        self._cmi = cmi
        self._C = C
        
        # Calculate the transconductance (C/4)
        transconductance = C / 4
        
        # CASE 1: Positive correlation (when cmi = 1)
        if cmi == 1:
            # Create a voltage-controlled current source with positive gain
            # Current flows from imi_pos to imi_neg proportional to Vmi*(C/4)
            self.VCCS('vccs', 'imi_pos', 'imi_neg', 'gnd', 'vmi', transconductance=transconductance)
        
        # CASE 2: No current (when cmi = 0)
        elif cmi == 0:
            # No current source needed, but we'll add a very high resistance path
            # to ensure the circuit is complete
            self.R('no_current', 'imi_pos', 'imi_neg', 1e9)  # Very high resistance
        
        # CASE 3: Negative correlation (when cmi = -1)
        elif cmi == -1:
            # Create a voltage-controlled current source with negative gain
            # Current flows from imi_pos to imi_neg proportional to -Vmi*(C/4)
            self.VCCS('vccs', 'imi_pos', 'imi_neg', 'vmi', 'gnd', transconductance=transconductance)


# Test the voltage-to-current converter
def main():
    # Create a circuit
    circuit = Circuit('Branch Current Test')
    
    # Add power supply
    vdd = 1
    circuit.V('dd', 'vdd', circuit.gnd, vdd@u_V)
    
    # Add input voltage source
    circuit.V('in', 'vmi', circuit.gnd, 0.5@u_V)  # Default input of 0.5V
    
    # Test with different cmi values
    results = []
    labels = []
    
    # Create three different current converters with different cmi values
    C = 1e-9
    converter1 = BranchCurrent(cmi=1, C=C)
    converter0 = BranchCurrent(cmi=0, C=C)
    converter_neg1 = BranchCurrent(cmi=-1, C=C)
    
    # Add the subcircuits to the circuit
    circuit.subcircuit(converter1)
    circuit.subcircuit(converter0)
    circuit.subcircuit(converter_neg1)
    
    # Add measurement resistors (1 ohm) to convert current to voltage for measurement
    # For proper current measurement, we need to measure the voltage drop across the resistor
    circuit.R('meas1', 'out1_high', circuit.gnd, 1)
    circuit.R('meas0', 'out0_high', circuit.gnd, 1)
    circuit.R('meas_neg1', 'out_neg1_high', circuit.gnd, 1)
    
    # Instantiate the converters
    # Connect imi_pos to the high side of the resistor and imi_neg to ground
    circuit.X('conv1', converter1.name, 'vmi', 'out1_high', circuit.gnd, 'vdd', circuit.gnd)
    circuit.X('conv0', converter0.name, 'vmi', 'out0_high', circuit.gnd, 'vdd', circuit.gnd)
    circuit.X('conv_neg1', converter_neg1.name, 'vmi', 'out_neg1_high', circuit.gnd, 'vdd', circuit.gnd)
    
    # Create simulator
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    
    # Run DC sweep analysis
    analysis = simulator.dc(Vin=slice(0, vdd, 0.01))
    
    # Extract results - measure voltage across 1 ohm resistors (equals current in amps)
    results = [
        (analysis['v-sweep'], analysis['out1_high']),
        (analysis['v-sweep'], analysis['out0_high']),
        (analysis['v-sweep'], analysis['out_neg1_high'])
    ]
    
    labels = [
        'cmi = 1 (Imi = Vmi*(C/4))',
        'cmi = 0 (Imi = 0)',
        'cmi = -1 (Imi = -Vmi*(C/4))'
    ]
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.title('Voltage-to-Current Converter Response')
    plt.xlabel('Input Voltage [V]')
    plt.ylabel('Output Current [A]')
    plt.grid(True)
    
    for i, (x, y) in enumerate(results):
        plt.plot(x, y, label=labels[i])
    
    # Add ideal response lines for comparison
    x = np.linspace(0, vdd, 101)
    # Calculate the scaling factor C/4
    scaling_factor = C / 4
    # Use the correct scaling factor for ideal response lines
    plt.plot(x, x * scaling_factor, 'k--', alpha=0.5, label=f'Ideal: I = V*(C/4) = V*{scaling_factor:.2e}')
    plt.plot(x, np.zeros_like(x), 'k--', alpha=0.5, label='Ideal: I = 0')
    plt.plot(x, -x * scaling_factor, 'k--', alpha=0.5, label=f'Ideal: I = -V*(C/4) = -V*{scaling_factor:.2e}')
    
    plt.legend()
    plt.tight_layout()

    # Save the plot to a PNG file
    plt.savefig('graphs/branch_current_simulation_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'branch_current_simulation_results.png'")

    plt.show()
    
    # Print some specific test points
    print("\nVoltage-to-Current Converter Verification:")
    
    test_voltages = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Test each cmi value
    for cmi_value in [1, 0, -1]:
        print(f"\ncmi = {cmi_value}:")
        
        # Test each voltage with a fresh circuit
        for test_v in test_voltages:
            # Create a new circuit for each test point
            test_circuit = Circuit(f'Branch Current Test (cmi={cmi_value}, v_in={test_v})')
            test_circuit.V('dd', 'vdd', test_circuit.gnd, vdd@u_V)
            test_circuit.V('in', 'vmi', test_circuit.gnd, test_v@u_V)
            
            # Create and add the converter subcircuit
            converter = BranchCurrent(cmi=cmi_value, C=C)
            test_circuit.subcircuit(converter)
            
            # Add measurement resistor (1 ohm) to convert current to voltage for measurement
            test_circuit.R('meas', 'out_high', test_circuit.gnd, 1)
            
            # Instantiate the converter with proper connections for current measurement
            test_circuit.X('conv', converter.name, 'vmi', 'out_high', test_circuit.gnd, 'vdd', test_circuit.gnd)
            
            # Run simulation
            test_simulator = test_circuit.simulator(temperature=25, nominal_temperature=25)
            analysis = test_simulator.operating_point()
            
            # Calculate ideal output current and error
            ideal_current = test_v * cmi_value * (C / 4)  # This directly implements Imi = cmi*Vmi*(C/4)
            
            # Extract output current (voltage across 1 ohm resistor equals current in amps)
            output_current = float(analysis['out_high'])
            error = abs(output_current - ideal_current)
            
            print(f"Input = {test_v:.2f}V => Output = {output_current:.3e}A (Ideal: {ideal_current:.3e}A, Error: {error:.3e}A)")

if __name__ == '__main__':
    # Set up logging
    logger = Logging.setup_logging()
    main()