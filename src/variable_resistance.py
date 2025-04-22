import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuitFactory
from PySpice.Unit import u_V, u_Ohm
import matplotlib.pyplot as plt
import numpy as np

from transmission_gate import TransmissionGate

class VariableResistance(SubCircuitFactory):
    """
    A variable resistance circuit using five transmission gates.
    
    The circuit takes two voltage inputs (Vi and Vi_bar) and two boolean control signals (Q and Q_bar),
    and provides a variable resistance between Vin and Vout terminals.
    
    Circuit connections:
    1. TG1: enable=Q, enable_bar=Q_bar, input=Vi, output=TG5.enable_bar
    2. TG2: enable=Q_bar, enable_bar=Q, input=Vi_bar, output=TG5.enable_bar
    3. TG3: enable=Q_bar, enable_bar=Q, input=Vi, output=TG5.enable
    4. TG4: enable=Q, enable_bar=Q_bar, input=Vi_bar, output=TG5.enable
    5. TG5: input=Vin, output=Vout
    
    Nodes:
    - vi: First control voltage input
    - vi_bar: Second control voltage input (complement of vi)
    - q: First boolean control input
    - q_bar: Second boolean control input (complement of q)
    - vin: Input signal terminal
    - vout: Output signal terminal
    - vdd: Power supply
    - gnd: Ground
    """
    NAME = 'variable_resistance'
    NODES = ('vi', 'vi_bar', 'q', 'q_bar', 'vin', 'vout', 'vdd', 'gnd')
    
    def __init__(self):
        super().__init__()
        
        # Add the transmission gate subcircuit
        transmission_gate = TransmissionGate()
        self.subcircuit(transmission_gate)
        
        # Create internal nodes for the connections
        tg5_enable = 'tg5_enable'
        tg5_enable_bar = 'tg5_enable_bar'
        
        # 1. First transmission gate: Q->Vi->TG5.enable_bar
        self.X('tg1', transmission_gate.NAME, 'vi', tg5_enable_bar, 'q', 'q_bar', 'vdd', 'gnd')
        
        # 2. Second transmission gate: Q_bar->Vi_bar->TG5.enable_bar
        self.X('tg2', transmission_gate.NAME, 'vi_bar', tg5_enable_bar, 'q_bar', 'q', 'vdd', 'gnd')
        
        # 3. Third transmission gate: Q_bar->Vi->TG5.enable
        self.X('tg3', transmission_gate.NAME, 'vi', tg5_enable, 'q_bar', 'q', 'vdd', 'gnd')
        
        # 4. Fourth transmission gate: Q->Vi_bar->TG5.enable
        self.X('tg4', transmission_gate.NAME, 'vi_bar', tg5_enable, 'q', 'q_bar', 'vdd', 'gnd')
        
        # 5. Fifth transmission gate: Vin->Vout controlled by the above gates
        self.X('tg5', transmission_gate.NAME, 'vin', 'vout', tg5_enable, tg5_enable_bar, 'vdd', 'gnd')


def test_variable_resistance():
    """
    Test the variable resistance circuit by measuring the resistance between Vin and Vout
    under different control conditions.
    """
    # Create a circuit
    circuit = Circuit('Variable Resistance Test')
    
    # Add power supply
    vdd = 1
    circuit.V('dd', 'vdd', circuit.gnd, vdd@u_V)
    
    # Add the variable resistance subcircuit
    variable_resistance = VariableResistance()
    circuit.subcircuit(variable_resistance)
    
    # Instantiate the variable resistance
    circuit.X('var_res', variable_resistance.NAME, 'vi', 'vi_bar', 'q', 'q_bar', 'vin', 'vout', 'vdd', circuit.gnd)
    
    # Add a load resistor to measure current
    R_load = 1e3  # 1kΩ
    circuit.R('load', 'vout', circuit.gnd, R_load@u_Ohm)
    
    # Add a fixed input voltage to Vin
    circuit.V('in', 'vin', circuit.gnd, 0.5@u_V)
    
    # Add control voltage sources once outside the loop
    q_src = circuit.V('q', 'q', circuit.gnd, 0@u_V)
    q_bar_src = circuit.V('q_bar', 'q_bar', circuit.gnd, 1@u_V)
    vi_src = circuit.V('vi', 'vi', circuit.gnd, 0@u_V)
    vi_bar_src = circuit.V('vi_bar', 'vi_bar', circuit.gnd, 1@u_V)
    
    # Create simulator
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    
    # Test cases: different combinations of control inputs
    test_cases = [
        {'q': 0, 'q_bar': 1, 'vi': 0, 'vi_bar': 1, 'desc': 'Q=0, Q_bar=1, Vi=0, Vi_bar=1'},
        {'q': 0, 'q_bar': 1, 'vi': 1, 'vi_bar': 0, 'desc': 'Q=0, Q_bar=1, Vi=1, Vi_bar=0'},
        {'q': 1, 'q_bar': 0, 'vi': 0, 'vi_bar': 1, 'desc': 'Q=1, Q_bar=0, Vi=0, Vi_bar=1'},
        {'q': 1, 'q_bar': 0, 'vi': 1, 'vi_bar': 0, 'desc': 'Q=1, Q_bar=0, Vi=1, Vi_bar=0'},
    ]
    
    # Arrays to store results
    control_configs = []
    resistances = []
    vout_values = []
    vin_values = []
    
    # Run tests for each configuration
    print("Testing variable resistance circuit with different control configurations:")
    
    for i, case in enumerate(test_cases):
        # Set control signals by updating the dc_value of existing sources
        q_src.dc_value = case['q']@u_V
        q_bar_src.dc_value = case['q_bar']@u_V
        vi_src.dc_value = case['vi']@u_V
        vi_bar_src.dc_value = case['vi_bar']@u_V
        
        # Run operating point analysis
        analysis = simulator.operating_point()
        
        # Get voltages - extract first element to avoid numpy deprecation warning
        vout = float(analysis['vout'][0])
        vin = 0.5  # Fixed input voltage
        
        # Calculate voltage drop and current
        voltage_drop = vin - vout
        current = vout / R_load
        
        # Calculate resistance
        epsilon = 1e-10  # Small value to avoid division by zero
        if current > epsilon:
            resistance = voltage_drop / current
        else:
            resistance = 1e6  # Very high resistance (effectively open circuit)
        
        # Store results
        control_configs.append(case['desc'])
        resistances.append(resistance)
        vout_values.append(vout)
        vin_values.append(vin)
        
        # Print results
        print(f"\nTest Case {i+1}: {case['desc']}")
        print(f"  Vout: {vout:.4f}V (with Vin: {vin:.4f}V)")
        print(f"  Resistance: {resistance:.2f}Ω")
        
    # Plot the results
    plt.figure(figsize=(10, 8))
    
    # Bar chart of resistances
    plt.subplot(2, 1, 1)
    plt.bar(control_configs, resistances)
    plt.ylabel('Resistance [Ω]')
    plt.title('Variable Resistance Circuit - Resistance vs Control Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    
    # Output voltages
    plt.subplot(2, 1, 2)
    plt.bar(control_configs, vout_values)
    for i, config in enumerate(control_configs):
        plt.annotate(f"{vout_values[i]:.4f}V", 
                     xy=(i, vout_values[i]), 
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center')
    
    plt.axhline(y=0.5, color='r', linestyle='--', label='Input Voltage (0.5V)')
    plt.ylabel('Output Voltage [V]')
    plt.title('Variable Resistance Circuit - Output Voltage vs Control Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('graphs/variable_resistance_test.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to 'variable_resistance_test.png'")
    
    plt.show()
    
    return resistances, vout_values


def sweep_test():
    """
    Test the variable resistance circuit by sweeping the input voltage and measuring
    the output under different control conditions.
    """
    # Create a circuit
    circuit = Circuit('Variable Resistance Sweep Test')
    
    # Add power supply
    vdd = 1
    circuit.V('dd', 'vdd', circuit.gnd, vdd@u_V)
    
    # Add the variable resistance subcircuit
    variable_resistance = VariableResistance()
    circuit.subcircuit(variable_resistance)
    
    # Instantiate the variable resistance
    circuit.X('var_res', variable_resistance.NAME, 'vi', 'vi_bar', 'q', 'q_bar', 'vin', 'vout', 'vdd', circuit.gnd)
    
    # Add a load resistor to measure current
    R_load = 1e3  # 1kΩ
    circuit.R('load', 'vout', circuit.gnd, R_load@u_Ohm)
    
    # Add input voltage source (will be swept)
    circuit.V('in', 'vin', circuit.gnd, 0@u_V)
    
    # Add control voltage sources once outside the loop
    q_src = circuit.V('q', 'q', circuit.gnd, 0@u_V)
    q_bar_src = circuit.V('q_bar', 'q_bar', circuit.gnd, 1@u_V)
    vi_src = circuit.V('vi', 'vi', circuit.gnd, 0@u_V)
    vi_bar_src = circuit.V('vi_bar', 'vi_bar', circuit.gnd, 1@u_V)
    
    # Create simulator
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    
    # Test cases: different combinations of control inputs
    test_cases = [
        {'q': 0, 'q_bar': 1, 'vi': 0, 'vi_bar': 1, 'desc': 'Q=0, Q_bar=1, Vi=0, Vi_bar=1'},
        {'q': 0, 'q_bar': 1, 'vi': 1, 'vi_bar': 0, 'desc': 'Q=0, Q_bar=1, Vi=1, Vi_bar=0'},
        {'q': 1, 'q_bar': 0, 'vi': 0, 'vi_bar': 1, 'desc': 'Q=1, Q_bar=0, Vi=0, Vi_bar=1'},
        {'q': 1, 'q_bar': 0, 'vi': 1, 'vi_bar': 0, 'desc': 'Q=1, Q_bar=0, Vi=1, Vi_bar=0'},
    ]
    
    # Define line styles, colors, and markers for better visibility
    line_styles = ['-', '--', '-.', ':']
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['o', 's', '^', 'x']
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    for i, case in enumerate(test_cases):
        # Set control signals by updating the dc_value of existing sources
        q_src.dc_value = case['q']@u_V
        q_bar_src.dc_value = case['q_bar']@u_V
        vi_src.dc_value = case['vi']@u_V
        vi_bar_src.dc_value = case['vi_bar']@u_V
        
        # Run DC sweep analysis
        analysis = simulator.dc(Vin=slice(0, vdd, 0.01))
        
        # Get voltages
        input_voltages = np.array(analysis['v-sweep'])
        output_voltages = np.array(analysis['vout'])
        
        # Calculate resistance at each point
        voltage_drops = input_voltages - output_voltages
        currents = output_voltages / R_load
        
        # Avoid division by zero with a small epsilon
        epsilon = 1e-10
        resistances = voltage_drops / (currents + epsilon)
        
        # Filter out extreme values for better visualization
        valid_indices = (currents > 1e-6) & (voltage_drops > 1e-6)
        if np.any(valid_indices):
            median_resistance = np.median(resistances[valid_indices])
        else:
            median_resistance = 1000
        
        # Use different line styles, colors, and markers for each case
        # For voltage transfer plot
        plt.subplot(2, 1, 1)
        plt.plot(input_voltages, output_voltages, 
                 linestyle=line_styles[i], 
                 color=colors[i], 
                 marker=markers[i], 
                 markevery=10,  # Place marker every 10 points to avoid overcrowding
                 linewidth=2,
                 label=case['desc'])
        
        # For resistance plot
        plt.subplot(2, 1, 2)
        plt.plot(input_voltages, resistances, 
                 linestyle=line_styles[i], 
                 color=colors[i], 
                 marker=markers[i], 
                 markevery=10,
                 linewidth=2,
                 label=f"{case['desc']} (Avg: {median_resistance:.2f}Ω)")
    
    # Add labels and legends
    plt.subplot(2, 1, 1)
    plt.plot([0, vdd], [0, vdd], 'k--', linewidth=1.5, label='Ideal (No Resistance)')
    plt.xlabel('Input Voltage [V]', fontsize=12)
    plt.ylabel('Output Voltage [V]', fontsize=12)
    plt.title('Variable Resistance Circuit - Voltage Transfer Characteristics', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10, loc='best')
    
    plt.subplot(2, 1, 2)
    plt.xlabel('Input Voltage [V]', fontsize=12)
    plt.ylabel('Resistance [Ω]', fontsize=12)
    plt.title('Variable Resistance Circuit - Resistance vs Input Voltage', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10, loc='best')
    
    # Add a vertical spacing between subplots
    plt.subplots_adjust(hspace=0.3)
    
    # Save the plot with higher resolution
    plt.savefig('graphs/variable_resistance_sweep.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'variable_resistance_sweep.png'")
    
    plt.show()


def sweep_vi_test():
    """
    Test how the resistance between Vin and Vout changes as a function of Vi,
    with separate curves for Q=0 and Q=1.
    """
    # Create a circuit
    circuit = Circuit('Variable Resistance Vi Sweep Test')
    
    # Add power supply
    vdd = 1
    circuit.V('dd', 'vdd', circuit.gnd, vdd@u_V)
    
    # Add the variable resistance subcircuit
    variable_resistance = VariableResistance()
    circuit.subcircuit(variable_resistance)
    
    # Instantiate the variable resistance
    circuit.X('var_res', variable_resistance.NAME, 'vi', 'vi_bar', 'q', 'q_bar', 'vin', 'vout', 'vdd', circuit.gnd)
    
    # Add a load resistor to measure current
    R_load = 1e3  # 1kΩ
    circuit.R('load', 'vout', circuit.gnd, R_load@u_Ohm)
    
    # Add a fixed input voltage to Vin (for measuring resistance)
    circuit.V('in', 'vin', circuit.gnd, 0.5@u_V)
    
    # Add control voltage sources
    q_src = circuit.V('q', 'q', circuit.gnd, 0@u_V)
    q_bar_src = circuit.V('q_bar', 'q_bar', circuit.gnd, 1@u_V)
    vi_src = circuit.V('vi', 'vi', circuit.gnd, 0@u_V)  # Will be swept
    vi_bar_src = circuit.V('vi_bar', 'vi_bar', circuit.gnd, 1@u_V)  # Will be complement of vi
    
    # Create simulator
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    
    # Store results for Q=0 and Q=1
    q_values = [0, 1]
    
    # Create non-uniform sweep with more points in the transition regions
    # Add more points near the edges (0-0.2 and 0.8-1.0) for better resolution
    vi_edges = np.linspace(0, 0.2, 31)  # More points in 0-0.2 range
    vi_middle = np.linspace(0.2, 0.8, 41)  # Regular points in the middle
    vi_edges2 = np.linspace(0.8, 1.0, 31)  # More points in 0.8-1.0 range
    vi_sweep = np.unique(np.concatenate([vi_edges, vi_middle, vi_edges2]))
    
    resistances = {q: np.zeros(len(vi_sweep)) for q in q_values}
    vout_values = {q: np.zeros(len(vi_sweep)) for q in q_values}
    
    # Fixed input voltage for resistance calculation
    vin_value = 0.5
    
    # Maximum resistance to cap the values (to avoid extreme peaks)
    max_resistance = 0.3e6  # 300kΩ cap
    
    # Test for each Q value
    for q in q_values:
        # Set Q and Q_bar
        q_src.dc_value = q@u_V
        q_bar_src.dc_value = (1-q)@u_V
        
        # Sweep Vi and calculate resistance at each point
        for i, vi in enumerate(vi_sweep):
            # Set Vi and Vi_bar (complement)
            vi_src.dc_value = vi@u_V
            vi_bar_src.dc_value = (vdd-vi)@u_V
            
            # Run operating point analysis
            analysis = simulator.operating_point()
            
            # Get output voltage
            vout = float(analysis['vout'])
            vout_values[q][i] = vout
            
            # Calculate resistance
            voltage_drop = vin_value - vout
            current = vout / R_load
            
            # Calculate resistance (avoid division by zero)
            epsilon = 1e-10
            if current > epsilon:
                res = voltage_drop / current
                # Cap the resistance to avoid extreme values
                resistances[q][i] = min(res, max_resistance)
            else:
                resistances[q][i] = max_resistance  # Cap very high resistance
    
    # Smooth the resistance curves to make transitions less abrupt
    # Apply a simple moving average filter
    window_size = 3
    for q in q_values:
        # Pad the array for the moving average
        padded = np.pad(resistances[q], (window_size//2, window_size//2), mode='edge')
        smoothed = np.zeros_like(resistances[q])
        for i in range(len(resistances[q])):
            # Calculate moving average
            smoothed[i] = np.mean(padded[i:i+window_size])
        resistances[q] = smoothed
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    # Define line styles for better visibility
    line_styles = ['-', '--']
    colors = ['blue', 'red']
    markers = ['o', 's']
    
    # Plot 1: Resistance vs Vi
    plt.subplot(2, 1, 1)
    for i, q in enumerate(q_values):
        plt.plot(vi_sweep, resistances[q], 
                 linestyle=line_styles[i], 
                 color=colors[i], 
                 marker=markers[i], 
                 markevery=10,
                 linewidth=2,
                 label=f'Q={q}')
    
    # Add a marker at Vi=0.2 to highlight the target resistance
    for i, q in enumerate(q_values):
        idx = np.abs(vi_sweep - 0.2).argmin()
        vi_at_0_2 = vi_sweep[idx]
        r_at_0_2 = resistances[q][idx]
        plt.plot(vi_at_0_2, r_at_0_2, 'ko', markersize=8)
        plt.annotate(f'R≈{r_at_0_2/1000:.1f}kΩ', 
                     xy=(vi_at_0_2, r_at_0_2),
                     xytext=(vi_at_0_2 + 0.05, r_at_0_2),
                     fontsize=10,
                     arrowprops=dict(arrowstyle='->'))
    
    plt.xlabel('Vi Voltage [V]', fontsize=12)
    plt.ylabel('Resistance [Ω]', fontsize=12)
    plt.title('Variable Resistance Circuit - Resistance vs Vi (Vi_bar = VDD - Vi, Vin=0.5V)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12, loc='best')
    
    # Plot 2: Output Voltage vs Vi
    plt.subplot(2, 1, 2)
    for i, q in enumerate(q_values):
        plt.plot(vi_sweep, vout_values[q], 
                 linestyle=line_styles[i], 
                 color=colors[i], 
                 marker=markers[i], 
                 markevery=10,
                 linewidth=2,
                 label=f'Q={q}')
    
    plt.axhline(y=vin_value, color='green', linestyle='-', label=f'Input Voltage ({vin_value}V)')
    plt.xlabel('Vi Voltage [V]', fontsize=12)
    plt.ylabel('Output Voltage [V]', fontsize=12)
    plt.title('Variable Resistance Circuit - Output Voltage vs Vi (Vin=0.5V)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12, loc='best')
    
    # Add a vertical spacing between subplots
    plt.subplots_adjust(hspace=0.3)
    
    # Save the plot
    plt.savefig('graphs/variable_resistance_vi_sweep.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'variable_resistance_vi_sweep.png'")
    
    plt.show()
    
    # Create log scale plot for resistance
    plt.figure(figsize=(12, 6))
    for i, q in enumerate(q_values):
        plt.semilogy(vi_sweep, resistances[q], 
                     linestyle=line_styles[i], 
                     color=colors[i], 
                     marker=markers[i], 
                     markevery=10,
                     linewidth=2,
                     label=f'Q={q}')
    
    plt.xlabel('Vi Voltage [V]', fontsize=12)
    plt.ylabel('Resistance [Ω] (log scale)', fontsize=12)
    plt.title('Variable Resistance Circuit - Resistance vs Vi (Log Scale)', fontsize=14)
    plt.grid(True, which="both")
    plt.legend(fontsize=12, loc='best')
    
    # Save the log-scale plot
    plt.savefig('graphs/variable_resistance_vi_sweep_log.png', dpi=300, bbox_inches='tight')
    print("Log-scale plot saved to 'variable_resistance_vi_sweep_log.png'")
    
    plt.show()


if __name__ == '__main__':
    # Set up logging
    logger = Logging.setup_logging()
    
    # Run tests
    test_variable_resistance()
    sweep_test()
    sweep_vi_test()  # Add the new Vi sweep test 