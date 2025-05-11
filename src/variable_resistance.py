import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import u_V, u_Ohm
import matplotlib.pyplot as plt
import numpy as np

from transmission_gate import TransmissionGate

class VariableResistance(SubCircuit):
    """
    A variable resistance circuit using five transmission gates.
    
    The circuit takes two voltage inputs (Vi and Vi_bar).
    and cmi (int): The ternary control value (-1, 0, or 1)
    and provides a variable resistance between Vin and Vout terminals.
    
    Circuit connections:
    1. TG1: enable=Q, enable_bar=Q_bar, input=Vi, output=TG5.enable_bar
    2. TG2: enable=Q_bar, enable_bar=Q, input=Vi_bar, output=TG5.enable_bar
    3. TG3: enable=Q_bar, enable_bar=Q, input=Vi, output=TG5.enable
    4. TG4: enable=Q, enable_bar=Q_bar, input=Vi_bar, output=TG5.enable
    5. TG5: input=Vin, output=Vout
    
    Nodes:
    - vi: First control voltage input
    - vin: Input signal terminal
    - vout: Output signal terminal
    - vdd: Power supply
    - gnd: Ground
    """
    NODES = ('vi', 'vin', 'vout', 'vdd', 'gnd')
    
    def __init__(self, cmi):
        super().__init__(f'VariableResistance_{cmi}', *self.NODES)
        
        # Add the transmission gate subcircuit
        transmission_gate = TransmissionGate()
        self.subcircuit(transmission_gate)
        # E <name> <out+> <out-> <in+> <in-> <gain>
        self.VCVS('inv', 'vi_bar', 'gnd', 'vdd', 'vi', 1)

        if cmi == 1:
            Vp = 'vi'
            Vn = 'vi_bar'
        elif cmi == -1:
            Vp = 'vi_bar'
            Vn = 'vi'
        else:
            raise ValueError("cmi must be 1 or -1 for now")

        self.X('tg', transmission_gate.name, 'vin', 'vout', Vn, Vp, 'vdd', 'gnd')


def sweep_vin_test():
    """
    Test the variable resistance circuit by sweeping the input voltage and measuring
    the output under different control conditions.
    """

    # Test cases: different combinations of control inputs
    test_cases = [
        {'cmi': -1, 'vi': 0, 'desc': 'cmi=-1, Vi=0'},
        {'cmi': -1, 'vi': 1, 'desc': 'cmi=-1, Vi=1'},
        {'cmi': 1, 'vi': 0, 'desc': 'cmi=1, Vi=0'},
        {'cmi': 1, 'vi': 1, 'desc': 'cmi=1, Vi=1'},
    ]

    # Define line styles, colors, and markers for better visibility
    line_styles = ['-', '--', '-.', ':']
    colors = ['blue', 'red', 'green', 'purple']
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    

    for i, case in enumerate(test_cases):
        # Create a circuit
        circuit = Circuit('Variable Resistance Sweep Test')
        
        # Add power supply
        vdd = 1
        circuit.V('dd', 'vdd', circuit.gnd, vdd@u_V)
        
        # Add the variable resistance subcircuit
        variable_resistance = VariableResistance(cmi=case['cmi'])
        circuit.subcircuit(variable_resistance)
        
        # Instantiate the variable resistance
        circuit.X('var_res', variable_resistance.name, 'vi', 'vin', 'vout', 'vdd', circuit.gnd)
        
        # Add a load resistor to measure current
        R_load = 1e3  # 1kΩ
        circuit.R('load', 'vout', circuit.gnd, R_load@u_Ohm)
        
        # Add input voltage source (will be swept)
        circuit.V('in', 'vin', circuit.gnd, 0@u_V)
        
        # Add control voltage sources once outside the loop
        circuit.V('vi', 'vi', circuit.gnd, case['vi']@u_V)
        
        # Create simulator
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        
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
                 markevery=10,  # Place marker every 10 points to avoid overcrowding
                 linewidth=2,
                 label=case['desc'])
        
        # For resistance plot
        plt.subplot(2, 1, 2)
        plt.plot(input_voltages, resistances, 
                 linestyle=line_styles[i], 
                 color=colors[i],
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
    # Store results for Q=0 and Q=1
    cmi_values = [-1, 1]
    
    # Create uniform sweep with more points in the transition regions
    vi_sweep = np.linspace(0, 1, 1001)
    
    resistances = {cmi: np.zeros(len(vi_sweep)) for cmi in cmi_values}
    vout_values = {cmi: np.zeros(len(vi_sweep)) for cmi in cmi_values}
    
    # Fixed input voltage for resistance calculation
    vin_value = 0.5
    
    # Test for each Q value
    for cmi in cmi_values:
        # Create a circuit
        circuit = Circuit('Variable Resistance Vi Sweep Test')
        
        # Add power supply
        vdd = 1
        circuit.V('dd', 'vdd', circuit.gnd, vdd@u_V)
        
        # Add the variable resistance subcircuit
        variable_resistance = VariableResistance(cmi=cmi)
        circuit.subcircuit(variable_resistance)
        
        # Instantiate the variable resistance
        circuit.X('var_res', variable_resistance.name, 'vi', 'vin', 'vout', 'vdd', circuit.gnd)
        
        # Add a load resistor to measure current
        R_load = 1e3  # 1kΩ
        circuit.R('load', 'vout', circuit.gnd, R_load@u_Ohm)
        
        # Add a fixed input voltage to Vin (for measuring resistance)
        circuit.V('in', 'vin', circuit.gnd, 0.5@u_V)
        
        # Add control voltage sources
        vi_src = circuit.V('vi', 'vi', circuit.gnd, 0@u_V)  # Will be swept
        
        # Create simulator
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        
        # Sweep Vi and calculate resistance at each point
        for i, vi in enumerate(vi_sweep):
            # Set Vi and Vi_bar (complement)
            vi_src.dc_value = vi@u_V
            
            # Run operating point analysis
            analysis = simulator.operating_point()
            
            # Get output voltage
            vout = float(analysis['vout'])
            vout_values[cmi][i] = vout
            
            # Calculate resistance
            voltage_drop = vin_value - vout
            current = vout / R_load
            
            # Calculate resistance
            res = voltage_drop / current
            resistances[cmi][i] = res
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    # Define line styles for better visibility
    line_styles = ['-', '--']
    colors = ['blue', 'red']

    print(f"Resistance (cmi=1, vi=0) = {resistances[1][0]/1000:.1f}kΩ")
    print(f"Resistance (cmi=1, vi=Vdd) = {resistances[1][-1]/1000:.1f}kΩ")
    print(f"Resistance (cmi=-1, vi=0) = {resistances[-1][0]/1000:.1f}kΩ")
    print(f"Resistance (cmi=-1, vi=Vdd) = {resistances[-1][-1]/1000:.1f}kΩ")
    
    # Plot 1: Resistance vs Vi
    plt.subplot(2, 1, 1)
    for i, cmi in enumerate(cmi_values):
        plt.plot(vi_sweep, resistances[cmi], 
                 linestyle=line_styles[i], 
                 color=colors[i],  
                 markevery=10,
                 linewidth=2,
                 label=f'cmi={cmi}')
    
    # Add a marker at Vi=0.2 to highlight the target resistance
    for i, cmi in enumerate(cmi_values):
        idx = np.abs(vi_sweep - 0.2).argmin()
        vi_at_0_2 = vi_sweep[idx]
        r_at_0_2 = resistances[cmi][idx]
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
    for i, cmi in enumerate(cmi_values):
        plt.plot(vi_sweep, vout_values[cmi], 
                 linestyle=line_styles[i], 
                 color=colors[i],
                 markevery=10,
                 linewidth=2,
                 label=f'cmi={cmi}')
    
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
    for i, cmi in enumerate(cmi_values):
        plt.semilogy(vi_sweep, resistances[cmi], 
                     linestyle=line_styles[i], 
                     color=colors[i],
                     markevery=10,
                     linewidth=2,
                     label=f'cmi={cmi}')
    
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
    sweep_vin_test()
    sweep_vi_test()