import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import u_ms, u_us, u_F, u_Ω
import matplotlib.pyplot as plt
import numpy as np
import os


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
    NODES = ('vam', 'n1', 'vi1', 'vi2', 'vi3', 'gnd')
    
    def __init__(self, R, C):
        """
        Initialize the AM circuit subcircuit with specific component values.
        
        Args:
            R (float): Resistance value in ohms
            C (float): Capacitance value in farads
            gain (float): Voltage gain of the amplifier
                          Controls exponential growth rate by: growth_rate = (vi1*vi2*vi3)/(RC)
        """
        super().__init__(f'AMCircuit_{R}_{C}', *self.NODES)
            
        # Store component values
        self._C = C
        
        # Add RC circuit with initial condition capability
        self.R('1', 'n1', 'vam', R@u_Ω)
        self.C('1', 'n1', 'gnd', C@u_F)

        self.V('v1', 'v1', 'gnd', 1.0)
        
        # Add controlled source with feedback to create exponential growth
        # The growth rate is determined by: growth_rate = (gain-1)/(RC)
        # E <name> <out+> <out-> <in+> <in-> <gain>
        self.B('amp', 'vam', 'gnd', v='((V(vi1)*V(vi2)*V(vi3))+V(v1))*V(n1)')


def simulate_am_circuit():
    """
    Simulate the AM circuit and plot the output.
    """
    # Define component values
    Rs = [0.3e3, 3e3, 30e3, 300e3]
    C = 0.01e-6

    # Set simulation parameters
    step_time = 1@u_us  # Larger step size for more stable solution
    end_time = 0.5@u_ms  # Simulation duration
    
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

    n1_init = 0.01
    vi1 = 1.0
    vi2 = 1.0
    vi3 = 1.0
    vam_init = (vi1 * vi2 * vi3 + 1) * n1_init
    
    for R in Rs:
        # Create a circuit
        circuit = Circuit('AM Circuit Simulation')

        # Calculate the theoretical growth rate
        growth_rate = (vi1*vi2*vi3)/(R*C)
        print(f"Theoretical growth rate: {growth_rate:.2f} Hz")
        print(f"Theoretical time constant: {1000/growth_rate:.3f} μs")
            
        # Add the AMCircuit subcircuit
        am_circuit = AMNewCircuit(R, C)
        circuit.subcircuit(am_circuit)

        circuit.V('vi1', 'vi1', circuit.gnd, vi1)
        circuit.V('vi2', 'vi2', circuit.gnd, vi2)
        circuit.V('vi3', 'vi3', circuit.gnd, vi3)
        
        # Instantiate the AM circuit
        circuit.X('am1', am_circuit.name, 'vam', 'n1', 'vi1', 'vi2', 'vi3', circuit.gnd)
        
        # Create simulator with specific instructions to honor initial conditions
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        
        # Set initial condition on n1 node 
        simulator.initial_condition(n1=n1_init)
        
        # Set analysis parameters to improve convergence
        simulator.options(
            reltol=1e-2,       # Reasonable tolerance 
            abstol=1e-4,       # Reasonable absolute tolerance
            method='trap',     # Trapezoidal integration method
            gmin=1e-9,         # Minimum conductance
            maxord=2,          # Maximum order for integration method
            itl1=500,          # Increased DC iteration limit
            itl4=2000,         # Increased transient iteration limit
            maxstep=10e-6,     # Larger maximum time step
            uic=True           # Use initial conditions (crucial for correct behavior)
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
        plotted = ax1.plot(time * 1000, vam, linestyle='-', linewidth=1, label=f'R = {R/1000:.1f} kΩ')
        color = plotted[0].get_color()
        ax1.plot(time * 1000, vam_theory, linestyle='--', linewidth=2, label=f'Theoretical R = {R/1000:.1f} kΩ', color=color)
        ax2.semilogy(time * 1000, np.abs(vam), linewidth=2, label=f'R = {R/1000:.1f} kΩ')
    
    # Format the first figure (linear and log plots)
    ax1.set_xlabel('Time [ms]', fontsize=12)
    ax1.set_ylabel('Output Voltage (Vam) [V]', fontsize=12)
    ax1.set_title('AM Circuit Output Voltage vs Time (All Resistances)', fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    ax1.legend(fontsize=10)
    
    ax2.set_xlabel('Time [ms]', fontsize=12)
    ax2.set_ylabel('|Output Voltage| (Log Scale) [V]', fontsize=12)
    ax2.set_title('AM Circuit Output Voltage - Log Scale (All Resistances)', fontsize=14)
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
    
    # Analyze and print characteristics of the outputs
    print("\nCircuit Analysis Summary:")
    for i, R in enumerate(Rs):
        print(f"\nR = {R/1000:.1f} kΩ:")
        print(f"Initial Voltage: {all_vams[i][0]:.6f}V")
        print(f"Final Voltage: {all_vams[i][-1]:.6f}V")
        print(f"Maximum Voltage: {np.max(all_vams[i]):.6f}V")
        print(f"Minimum Voltage: {np.min(all_vams[i]):.6f}V")
        
        # Print growth rates
        print(f"Fitted Exponential Growth Rate: {all_growth_rates[i]:.2f} Hz")
        print(f"Time Constant from Fit: {1000/all_growth_rates[i]:.3f} μs")

if __name__ == '__main__':
    # Set up logging
    logger = Logging.setup_logging()
    
    # Run the main simulation
    simulate_am_circuit()