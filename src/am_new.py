import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import u_ms, u_us, u_F, u_Ω, u_V
import matplotlib.pyplot as plt
import numpy as np
import os
from and_circuit import AndGate
from branch_voltage import TernaryMultiplexer

class AMNewCircuit(SubCircuit):
    """
    AM Circuit subcircuit that implements a positive feedback oscillator.
    
    This circuit uses a voltage-controlled voltage source with positive feedback
    to create an ideal exponential growth circuit with precise control over growth rate.
    
    Nodes:
    - vam: Output voltage of the circuit
    - n1: Node between the two branches
    - vi1: Voltage of first variable
    - vi2: Voltage of second variable
    - vi3: Voltage of third variable
    - gnd: Ground reference
    """
    NODES = ('vam', 'n1', 'vi1', 'vi2', 'vi3', 'vdd', 'gnd')
    
    def __init__(self, cm1, cm2, cm3, R, C):
        """
        Initialize the AM circuit subcircuit with specific component values.
        
        Args:
            R (float): Resistance value in ohms
            C (float): Capacitance value in farads
            gain (float): Voltage gain of the amplifier
                          Controls exponential growth rate by: growth_rate = (vi1*vi2*vi3)/(RC)
        """
        super().__init__(f'AMCircuit_{cm1}_{cm2}_{cm3}_{R}_{C}', *self.NODES)
        
        # # Create and add the ternary multiplexers
        mux1 = TernaryMultiplexer(cmi=-cm1)
        mux2 = TernaryMultiplexer(cmi=-cm2)
        mux3 = TernaryMultiplexer(cmi=-cm3)
        
        # # Create and add the NOR gate
        nor_gate = AndGate()
        
        # Add the subcircuits
        self.subcircuit(mux1)
        self.subcircuit(mux2)
        self.subcircuit(mux3)
        self.subcircuit(nor_gate)

        mux1_out = 'mux1_out'
        mux2_out = 'mux2_out'
        mux3_out = 'mux3_out'
        nor12_out = 'nor12_out'
        nor123_out = 'nor123_out'
        gain_node = 'gain_node'
        
        # Instantiate the multiplexers
        self.X('mux1', mux1.name, 'vi1', mux1_out, 'vdd', 'gnd')
        self.X('mux2', mux2.name, 'vi2', mux2_out, 'vdd', 'gnd')
        self.X('mux3', mux3.name, 'vi3', mux3_out, 'vdd', 'gnd')

        # Instantiate the NOR gate to compute V(vi1)*V(vi2)*V(vi3)
        self.X('nor12', nor_gate.name, mux1_out, mux2_out, nor12_out, 'vdd', 'gnd')
        self.X('nor123', nor_gate.name, nor12_out, mux3_out, nor123_out, 'vdd', 'gnd')
        # self.B('nor123', nor123_out, 'gnd', v='V(mux1_out)*V(mux2_out)*V(mux3_out)')

        # Add 1V power supply to get V(vi1)*V(vi2)*V(vi3) + 1V
        self.V('gain_node', gain_node, nor123_out, 1@u_V)
        
        # Add RC circuit with initial condition capability
        self.R('1', 'n1', 'vam', R@u_Ω)
        # variable_resistance = VariableResistance(cmi=-1)
        # self.subcircuit(variable_resistance)
        # self.X('var_res', variable_resistance.name, gain_node, 'n1', 'vam', 'vdd', 'gnd')
        
        self.C('1', 'n1', 'gnd', C@u_F)
        
        # Add controlled source with feedback to create exponential growth
        # The growth rate is determined by: growth_rate = (gain-1)/(RC)
        # E <name> <out+> <out-> <in+> <in-> <gain>
        self.B('amp', 'vam', 'gnd', v='V(gain_node)*V(n1)')
        # self.VCVS('amp', 'vam', 'gnd', 'n1', 'gnd', voltage_gain=2)


def simulate_am_circuit():
    """
    Simulate the AM circuit and plot the output.
    """
    # Define component values
    # R = 10e3

    R = 15e3  # 15kΩ
    C = 10e-9  # 10nF

    # Set simulation parameters
    step_time = 1@u_us  # Larger step size for more stable solution
    end_time = 500@u_us  # Simulation duration
    
    # Create figures for combined plots
    fig1 = plt.figure(figsize=(15, 10))
    ax1 = fig1.add_subplot(2, 1, 1)
    ax2 = fig1.add_subplot(2, 1, 2)
    
    # Lists to store data for analysis
    all_times = []
    all_vams = []
    all_theories = []
    all_growth_rates = []
    all_labels = []

    n1_init = 1.0
    cases = [
        {'vi1': 0.0, 'vi2': 0.0, 'vi3': 0.0, 'cm1': 1, 'cm2': 1, 'cm3': 1},
        {'vi1': 0.0, 'vi2': 0.0, 'vi3': 0.0, 'cm1': -1, 'cm2': -1, 'cm3': -1},
        {'vi1': 1.0, 'vi2': 1.0, 'vi3': 1.0, 'cm1': 1, 'cm2': 1, 'cm3': 1},
        {'vi1': 1.0, 'vi2': 1.0, 'vi3': 1.0, 'cm1': -1, 'cm2': -1, 'cm3': -1},
    ]
    
    ls=[(5,(5, 5)), (5,(5, 5)),(0,(5, 5)),(0,(5, 5))]
    for i, case in enumerate(cases):
        vi1 = case['vi1']
        vi2 = case['vi2']
        vi3 = case['vi3']
        cm1 = case['cm1']
        cm2 = case['cm2']
        cm3 = case['cm3']

        # Create a circuit
        circuit = Circuit('AM Circuit Simulation')

        mux1 = 1 - vi1 if cm1 == 1 else vi1
        mux2 = 1 - vi2 if cm2 == 1 else vi2
        mux3 = 1 - vi3 if cm3 == 1 else vi3
        prod = mux1 * mux2 * mux3

        # Calculate the theoretical growth rate
        vam_init = (prod + 1) * n1_init
        growth_rate = prod/(R*C)
        print(f"Theoretical growth rate: {growth_rate:.2f} Hz")
        if growth_rate > 0:
            print(f"Theoretical time constant: {1000/growth_rate:.3f} μs")
        else:
            print("Growth rate is zero")
            
        # Add the AMCircuit subcircuit
        am_circuit = AMNewCircuit(cm1, cm2, cm3, R, C)
        circuit.subcircuit(am_circuit)

        circuit.V('vdd', 'vdd', circuit.gnd, 1@u_V)

        circuit.V('vi1', 'vi1', circuit.gnd, vi1)
        circuit.V('vi2', 'vi2', circuit.gnd, vi2)
        circuit.V('vi3', 'vi3', circuit.gnd, vi3)
        
        # Instantiate the AM circuit
        circuit.X('am1', am_circuit.name, 'vam', 'n1', 'vi1', 'vi2', 'vi3', 'vdd', circuit.gnd)
        
        # Create simulator with specific instructions to honor initial conditions
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        
        # Set initial condition on n1 node 
        simulator.initial_condition(n1=n1_init)
        
        # Set analysis parameters to improve convergence
        simulator.options(
            reltol=1e-2,     # Relaxed relative tolerance (was 1e-3)
            abstol=1e-5,     # Relaxed absolute tolerance (was 1e-6)
            itl1=500,        # Increased DC iteration limit (was 100)
            itl2=200,        # Increased DC transfer curve iteration limit (was 50)
            itl4=100,        # Transient analysis iteration limit
            gmin=1e-10,      # Minimum conductance
            method='gear',   # Integration method (alternatives: 'trap', 'euler')
            maxord=2         # Maximum order for integration method
        )
        
        # Run transient simulation
        print(f"Simulating AM circuit with R = {R/1000:.1f} kΩ...")
        analysis = simulator.transient(step_time=step_time, end_time=end_time)
        
        # Extract time and output voltage
        time = np.array(analysis.time)
        vam = np.array(analysis['vam'])
        
        # Calculate exponential growth rate directly from the simulation data
        # First, find a section with clear exponential growth
        log_vam = np.abs(vam)
        log_vam[log_vam < 1e-6] = 1e-6  # Replace very small values
        
        # Skip early points for fitting (to avoid initial transients)
        skip_points = int(len(time) * 0.05)  # Skip first 5% of points
        if skip_points < 5:
            skip_points = 5  # Ensure we skip at least 5 points
        
        # Use at least 100 points for fitting
        fit_points = 100
        if len(time) - skip_points < fit_points:
            fit_points = len(time) - skip_points
            if fit_points < 10:
                fit_points = len(time) - 1  # Use all but first point as last resort
    
        # Calculate theoretical curve using the extracted growth rate
        vam_theory = vam_init * np.exp(growth_rate * time)
        
        # Store data for later analysis
        all_times.append(time)
        all_vams.append(vam)
        all_theories.append(vam_theory)
        all_growth_rates.append(growth_rate)
        all_labels.append(f'R = {R/1000:.1f} kΩ')
    
        # Plot on the combined figures
        label = f'$V_{{i_1}}={vi1}$ V, $V_{{i_2}}={vi2}$ V, $V_{{i_3}}={vi3}$ V, $c_{{m,i_1}}={cm1}$, $c_{{m,i_2}}={cm2}$, $c_{{m,i_3}}={cm3}$'
        plotted = ax1.plot(time * 1000000, vam, linestyle=ls[i], linewidth=2, label=label)  # noqa: F841
        # color = plotted[0].get_color()
        # ax1.plot(time * 1000000, vam_theory, linestyle='--', linewidth=2, label=f'Theoretical vi1={vi1}, vi2={vi2}, vi3={vi3}, cm1={cm1}, cm2={cm2}, cm3={cm3}', color=color)
        ax2.semilogy(time * 1000000, np.abs(vam), linestyle=ls[i], linewidth=2, label=label)
    
    # Format the first figure (linear and log plots)
    ax1.set_xlabel('Time [$\mu$s]', fontsize=12)
    ax1.set_ylabel('Output Voltage (Vam) [V]', fontsize=12)
    ax1.set_title('$V_{{a_m}}$ vs Time', fontsize=14)
    ax1.set_ylim(0, 20)
    ax1.grid(True)
    ax1.legend(fontsize=10)
    
    ax2.set_xlabel('Time [$\mu$s]', fontsize=12)
    ax2.set_ylabel('|Output Voltage| (Log Scale) [V]', fontsize=12)
    ax2.set_title('$V_{{a_m}}$ vs Time - Log Scale', fontsize=14)
    ax2.grid(True)
    ax2.legend(fontsize=10)
    
    fig1.tight_layout()
    
    # Ensure directory exists
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
        
    # Save the plots
    fig1.savefig('graphs/am_new_output.png', dpi=300, bbox_inches='tight')
    print("Combined plot saved to 'graphs/am_new_output.png'")
    
    print("Combined logarithmic plot saved to 'graphs/am_new_output.png'")
    
    plt.show()

if __name__ == '__main__':
    # Set up logging
    logger = Logging.setup_logging()
    
    # Run the main simulation
    simulate_am_circuit()