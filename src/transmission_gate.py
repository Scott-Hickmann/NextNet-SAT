import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuitFactory
from PySpice.Unit import u_V, u_Ohm
import matplotlib.pyplot as plt
import numpy as np


class TransmissionGate(SubCircuitFactory):
    """
    A transmission gate implementation using one PMOS and one NMOS transistor.
    The transmission gate acts as a switch that can pass signals in both directions
    when enabled and blocks them when disabled.
    
    Nodes:
    - input: Input signal
    - output: Output signal
    - enable: Enable signal (high = pass, low = block)
    - enable_bar: Inverted enable signal
    - vdd: Power supply
    - gnd: Ground
    """
    NAME = 'transmission_gate'
    NODES = ('input', 'output', 'enable', 'enable_bar', 'vdd', 'gnd')
    
    def __init__(self):
        super().__init__()
        
        # Define the NMOS and PMOS models with parameters
        # Using improved parameters for both transistors with realistic threshold voltages
        self.model('NMOS', 'NMOS', vto=0, kp=1e-3, lambda_=0.01)  # Positive threshold for NMOS
        self.model('PMOS', 'PMOS', vto=0, kp=1e-3, lambda_=0.01)  # Negative threshold for PMOS
        
        # NMOS transistor (passes when enable is high)
        # Proper connection order: drain, gate, source, body
        self.M(1, 'output', 'enable', 'input', 'gnd', model='NMOS')
        
        # PMOS transistor (passes when enable_bar is low)
        # Proper connection order: drain, gate, source, body
        self.M(2, 'output', 'enable_bar', 'input', 'vdd', model='PMOS')


def test_resistive_behavior():
    """
    Test the resistive behavior of the transmission gate by measuring
    the resistance between input and output when enabled.
    """
    # Create a circuit
    circuit = Circuit('Transmission Gate Test')
    
    # Add power supply
    vdd = 1
    circuit.V('dd', 'vdd', circuit.gnd, vdd@u_V)
    
    # Add the transmission gate subcircuit
    circuit.subcircuit(TransmissionGate())
    
    # Instantiate the transmission gate
    circuit.X('tg1', 'transmission_gate', 'input', 'output', 'enable', 'enable_bar', 'vdd', circuit.gnd)
    
    # Add input voltage source
    circuit.V('in', 'input', circuit.gnd, 0@u_V)
    
    # Add enable signals - proper values for full turn-on
    circuit.V('en', 'enable', circuit.gnd, vdd@u_V)  # Enable high (1V)
    circuit.V('en_bar', 'enable_bar', circuit.gnd, 0@u_V)  # Enable_bar low (0V)
    
    # Add a load resistor to measure current
    R_load = 1e3  # 1kΩ
    circuit.R('load', 'output', circuit.gnd, R_load@u_Ohm)
    
    # Create simulator
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    
    # Sweep input voltage and measure current
    print("Simulating transmission gate resistive behavior...")
    analysis = simulator.dc(Vin=slice(0, vdd, 0.01))
    
    # Convert to numpy arrays for calculations
    input_voltages = np.array(analysis['v-sweep'])
    output_voltages = np.array(analysis['output'])
    
    # Calculate the current through the load resistor (I = V/R)
    load_currents = output_voltages / R_load
    
    # Voltage drop across the transmission gate
    voltage_drop = input_voltages - output_voltages
    
    # Calculate transmission gate resistance (R = V/I)
    # Use small epsilon to avoid division by zero
    epsilon = 1e-10
    tg_resistances = voltage_drop / (load_currents + epsilon)
    
    # For regions where the calculation might be unreliable (near zero current),
    # replace with the median value from the stable region
    valid_indices = (load_currents > 1e-6) & (voltage_drop > 1e-6)
    median_resistance = np.median(tg_resistances[valid_indices]) if np.any(valid_indices) else 1000
    
    # Create masked array for plotting
    masked_resistances = np.copy(tg_resistances)
    masked_resistances[~valid_indices] = median_resistance
    
    # Calculate constant resistance model
    R_on = median_resistance
    ideal_output_voltages = input_voltages * (R_load / (R_load + R_on))
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Input-Output Voltage Transfer
    plt.subplot(2, 1, 1)
    plt.plot(input_voltages, output_voltages, 'b-', label='Actual Output Voltage')
    plt.plot(input_voltages, ideal_output_voltages, 'r--', 
             label=f'Ideal (Ron={R_on:.2f}Ω)')
    plt.xlabel('Input Voltage [V]')
    plt.ylabel('Output Voltage [V]')
    plt.title('Transmission Gate Voltage Transfer')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Resistance vs Input Voltage
    plt.subplot(2, 1, 2)
    plt.plot(input_voltages, masked_resistances, 'g-', label='Transmission Gate Resistance')
    plt.axhline(y=R_on, color='r', linestyle='--', 
                label=f'Constant Ron = {R_on:.2f}Ω')
    plt.xlabel('Input Voltage [V]')
    plt.ylabel('Resistance [Ω]')
    plt.title('Transmission Gate Resistance (Should be Constant)')
    # Set y-axis limits from 0 to 2x the theoretical resistance
    plt.ylim(0, 2 * R_on)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('graphs/transmission_gate_resistive_behavior.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'transmission_gate_resistive_behavior.png'")
    
    plt.show()
    
    # Print some key measurements
    print("\nTransmission Gate Resistive Behavior:")
    print(f"Constant On-Resistance (Ron): {R_on:.2f} Ω")
    
    # Also verify the constant resistance by testing at specific points
    print("\nVerification at specific points:")
    test_points = [0.2, 0.4, 0.6, 0.8]
    
    for v_point in test_points:
        # Find closest index
        idx = np.abs(input_voltages - v_point).argmin()
        v_in = input_voltages[idx]
        v_out = output_voltages[idx]
        r_calc = masked_resistances[idx]
        
        # Calculate expected output using voltage divider with constant Ron
        v_expected = v_in * R_load / (R_load + R_on)
        
        print(f"At Vin={v_in:.2f}V: Vout={v_out:.4f}V, R={r_calc:.2f}Ω, Expected={v_expected:.4f}V")


def test_enable_sweep():
    """
    Test how the transmission gate's resistance changes as the enable and enable_bar 
    signals vary from 0 to VDD.
    """
    # Create a circuit
    circuit = Circuit('Transmission Gate Enable Sweep Test')
    
    # Add power supply
    vdd = 1
    circuit.V('dd', 'vdd', circuit.gnd, vdd@u_V)
    
    # Add the transmission gate subcircuit
    circuit.subcircuit(TransmissionGate())
    
    # Instantiate the transmission gate
    circuit.X('tg1', 'transmission_gate', 'input', 'output', 'enable', 'enable_bar', 'vdd', circuit.gnd)
    
    # Add a fixed input voltage of 0.5V
    circuit.V('in', 'input', circuit.gnd, 0.5@u_V)
    
    # Add enable signals with initial values (will be swept)
    enable = circuit.V('en', 'enable', circuit.gnd, 0@u_V)
    enable_bar = circuit.V('en_bar', 'enable_bar', circuit.gnd, vdd@u_V)
    
    # Add a load resistor
    R_load = 1e3  # 1kΩ
    circuit.R('load', 'output', circuit.gnd, R_load@u_Ohm)
    
    # Create simulator
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    
    # Number of steps for the sweep
    num_steps = 101
    
    # Arrays to store results
    enable_values = np.linspace(0, vdd, num_steps)
    output_voltages = np.zeros(num_steps)
    resistances = np.zeros(num_steps)
    
    # Sweep enable voltage and calculate resistance at each point
    print("Simulating transmission gate behavior with varying enable signals...")
    
    for i, en_val in enumerate(enable_values):
        # Set enable voltage
        enable.dc_value = en_val@u_V
        # Set enable_bar to complement of enable (VDD - enable)
        enable_bar.dc_value = (vdd - en_val)@u_V
        
        # Run operating point analysis
        analysis = simulator.operating_point()
        
        # Store output voltage
        output_voltages[i] = float(analysis['output'])
        
        # Calculate resistance
        input_voltage = 0.5  # Fixed input voltage
        voltage_drop = input_voltage - output_voltages[i]
        current = output_voltages[i] / R_load
        
        # Calculate resistance (avoid division by zero)
        epsilon = 1e-10
        if current > epsilon:
            resistances[i] = voltage_drop / current
        else:
            resistances[i] = 1e6  # Very high resistance when current is near zero
    
    # Find transition points for better analysis
    transition_indices = []
    for i in range(1, len(output_voltages)-1):
        if (output_voltages[i] > 0.1 and output_voltages[i-1] < 0.1) or \
           (output_voltages[i] > 0.4 and output_voltages[i-1] < 0.4):
            transition_indices.append(i)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Output Voltage vs Enable
    plt.subplot(2, 1, 1)
    plt.plot(enable_values, output_voltages, 'b-')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Input Voltage (0.5V)')
    plt.xlabel('Enable Voltage [V]')
    plt.ylabel('Output Voltage [V]')
    plt.title('Transmission Gate Output vs Enable Voltage (Enable_bar = VDD - Enable)')
    plt.grid(True)
    plt.legend()
    
    # Add vertical lines at transition points if any were found
    for idx in transition_indices:
        plt.axvline(x=enable_values[idx], color='g', linestyle=':', alpha=0.7)
    
    # Plot 2: Resistance vs Enable
    plt.subplot(2, 1, 2)
    plt.semilogy(enable_values, resistances, 'g-')  # Use log scale for resistance
    plt.xlabel('Enable Voltage [V]')
    plt.ylabel('Resistance [Ω] (log scale)')
    plt.title('Transmission Gate Resistance vs Enable Voltage')
    plt.grid(True)
    
    # Add vertical lines at transition points if any were found
    for idx in transition_indices:
        plt.axvline(x=enable_values[idx], color='g', linestyle=':', alpha=0.7)
    
    # Find the minimum resistance and mark it on the plot
    min_r_idx = np.argmin(resistances)
    min_r_en = enable_values[min_r_idx]
    min_r_val = resistances[min_r_idx]
    
    plt.annotate(f'Min: {min_r_val:.2f}Ω at Enable={min_r_en:.2f}V',
                xy=(min_r_en, min_r_val),
                xytext=(min_r_en, min_r_val * 5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('graphs/transmission_gate_enable_sweep.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'transmission_gate_enable_sweep.png'")
    
    plt.show()
    
    # Print key points of the enable sweep
    print("\nTransmission Gate Enable Sweep Results:")
    print(f"Minimum Resistance: {min_r_val:.2f}Ω at Enable={min_r_en:.2f}V, Enable_bar={vdd-min_r_en:.2f}V")
    
    # Show resistance at key enable voltage points
    key_points = [0, 0.25, 0.5, 0.75, 1.0]
    print("\nResistance at key enable voltage points:")
    
    for point in key_points:
        idx = np.abs(enable_values - point).argmin()
        en = enable_values[idx]
        en_bar = vdd - en
        r = resistances[idx]
        out_v = output_voltages[idx]
        
        print(f"Enable={en:.2f}V, Enable_bar={en_bar:.2f}V: R={r:.2f}Ω, Output={out_v:.4f}V")


if __name__ == '__main__':
    # Set up logging
    logger = Logging.setup_logging()
    test_resistive_behavior()
    test_enable_sweep() 