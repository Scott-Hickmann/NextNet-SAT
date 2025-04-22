import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import u_V, u_kΩ, u_Ω, u_ms, u_us, u_F
import matplotlib.pyplot as plt
import numpy as np
import os
import mpmath as mp

from opamp import BasicOperationalAmplifier

def effective_gain(A0, wp, R, C):
    """return λ and A_eff solved from λ = (A(λ)-1)/(RC)"""
    def eqn(lam):
        return A0/(1 + lam/wp) - 1 - lam*R*C
    lam = mp.findroot(eqn, 1/(R*C))      # good initial guess
    return float(lam), float(lam*R*C)


class AMCircuit(SubCircuit):
    """
    AM Circuit subcircuit that implements a positive feedback oscillator.
    
    This circuit uses an operational amplifier with two capacitors of capacitance C,
    but exposes nodes for external resistive components to be connected.
    
    Nodes:
    - vout: Output voltage of the circuit
    - r1_a: First terminal for external resistive component 1 (connected to vout)
    - r1_b: Second terminal for external resistive component 1 (connected to op_plus)
    - r2_a: First terminal for external resistive component 2 (connected to op_minus)
    - r2_b: Second terminal for external resistive component 2 (connected to gnd)
    - vdd: Positive power supply
    - vss: Negative power supply
    - gnd: Ground reference
    - op_plus: Non-inverting input of the op-amp (exposed for measurement)
    - op_minus: Inverting input of the op-amp (exposed for measurement)
    """
    NODES = ('vout', 'r1_a', 'r1_b', 'r2_a', 'r2_b', 'vdd', 'vss', 'gnd', 'op_plus', 'op_minus')
    
    def __init__(self, C=0.01e-6):
        """
        Initialize the AM circuit subcircuit with specific capacitance value.
        
        Args:
            C (float): Capacitance value in farads (default: 0.01μF)
        """
        super().__init__(f'AMCircuit_{C}', *self.NODES)
            
        # Store component values
        self._C = C
        
        # Add the opamp subcircuit
        self.subcircuit(BasicOperationalAmplifier())
        
        # Connect vout to r1_a (to be connected to external resistive component)
        self.R('vout_to_r1a', 'vout', 'r1_a', 0.001@u_Ω)  # Very small resistor as a wire
        
        # Connect r1_b to op_plus
        self.R('r1b_to_op_plus', 'r1_b', 'op_plus', 0.001@u_Ω)  # Very small resistor as a wire
        
        # Connect r2_a to op_minus
        self.R('r2a_to_op_minus', 'r2_a', 'op_minus', 0.001@u_Ω)  # Very small resistor as a wire

        # Connect r2_b to gnd
        self.R('r2b_to_gnd', 'r2_b', 'gnd', 0.001@u_Ω)  # Very small resistor as a wire
        
        # Add capacitors
        self.C('1', 'op_minus', 'gnd', C@u_F)
        self.C('2', 'op_plus', 'gnd', C@u_F)
        
        # Instantiate the opamp
        self.X('opamp', 'BasicOperationalAmplifier', 
              'op_plus', 'op_minus', 'vout', 'vdd', 'vss')


def simulate_am_circuit():
    """
    Simulate the AM circuit and plot the output.
    """
    # Create a circuit
    circuit = Circuit('AM Circuit Simulation')
    
    # Define component values
    R = 100e3
    C = 0.01e-6
    
    # Add power supplies
    Vdd = 1.0
    Vss = -1.0
    circuit.V('dd', 'vdd', circuit.gnd, Vdd@u_V)
    circuit.V('ss', 'vss', circuit.gnd, Vss@u_V)
    
    # Create the AM circuit subcircuit
    am = AMCircuit(C=C)
    circuit.subcircuit(am)
    
    # Instantiate the AM circuit
    circuit.X('am', am.name, 'vam', 'r1_a', 'r1_b', 'r2_a', 'r2_b', 'vdd', 'vss', circuit.gnd, 'op_plus', 'op_minus')
    
    # Add external resistive components (standard resistors in this case)
    circuit.R('1', 'r1_a', 'r1_b', R@u_Ω)  # External resistor between vout and op_plus
    circuit.R('2', 'r2_a', 'r2_b', R@u_Ω)  # External resistor between op_minus and gnd
    
    # Add initial condition to kickstart oscillation
    circuit.PulseVoltageSource('kick', 'kick', circuit.gnd,
                               initial_value=0@u_V,
                               pulsed_value=0.1@u_V,
                               pulse_width=10@u_us,
                               period=1@u_ms,
                               delay_time=0@u_us,
                               rise_time=1@u_us,
                               fall_time=1@u_us)
    
    # Connect the kick source to the op_plus node
    circuit.R('kick', 'kick', 'op_plus', 1@u_kΩ)
    
    # Create simulator
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    
    # Set simulation parameters - shorter time to see exponential growth
    step_time = 0.1@u_us   # Finer resolution
    end_time = 0.5@u_ms    # Shorter time to focus on initial growth phase
    
    # Set analysis parameters to improve convergence
    simulator.options(reltol=1e-3, abstol=1e-6, method='gear')
    
    # Run transient simulation
    print("Simulating AM circuit...")
    analysis = simulator.transient(step_time=step_time, end_time=end_time)
    
    # Extract time and output voltage
    time = np.array(analysis.time)
    vam = np.array(analysis['vam'])
    op_plus = np.array(analysis['op_plus'])  # Now accessing the direct node
    op_minus = np.array(analysis['op_minus'])  # Now accessing the direct node
    
    # Calculate exponential growth rate directly from the simulation data
    # First, find a section with clear exponential growth
    log_vam = np.abs(vam)
    log_vam[log_vam < 1e-6] = 1e-6  # Replace very small values
    
    # Take logarithm to linearize the exponential growth
    log_vals = np.log(log_vam)
    
    # Find the linear segment (where exponential growth is steady)
    # Use points from when growth becomes noticeable until it gets too large
    threshold_low = 10.0  # Voltage where growth is clearly established
    threshold_high = 1e8  # Upper limit for analysis before numerical issues
    
    valid_indices = np.where((log_vam > threshold_low) & (log_vam < threshold_high))[0]
    
    # Extract time and log-voltage for the valid range
    t_valid = time[valid_indices]
    log_v_valid = log_vals[valid_indices]
    
    # Linear fit to the log data
    from scipy.stats import linregress
    slope, intercept, r_value, _, _ = linregress(t_valid, log_v_valid)
    
    # Extract growth rate from the slope
    growth_rate = slope
    v0 = np.exp(intercept)  # Initial value at t=0
    
    print(f"Growth rate from linear fit: {growth_rate:.2f} Hz")
    print(f"Initial amplitude from fit: {v0:.6f}")
    print(f"R-squared of linear fit: {r_value**2:.6f}")
    
    # Calculate theoretical curve using the extracted growth rate
    vam_theory = v0 * np.exp(growth_rate * time)
    
    # Compare with formula-based estimate
    # For positive feedback amplifier, theoretical growth rate approximation
    wp = 2*np.pi*100 # rad/s   (pole frequency in the opamp sub‑circuit)
    A0 = 1e5 # DC open‑loop gain
    lam, Aeff = effective_gain(A0, wp, R, C)
    lam_Hz = lam / (2 * np.pi)
    
    theoretical_growth_rate = lam_Hz
    print(f"Theoretical growth rate: {R} {C} {R * C} {theoretical_growth_rate:.2f} Hz")
    
    # Create an alternative theoretical curve using the formula-based estimate
    # Use the same starting point as the fitted model
    alt_theory = v0 * np.exp(theoretical_growth_rate * time)

    # Limit the theoretical curves to avoid overflow
    vam_theory = np.clip(vam_theory, -1e12, 1e12)
    if 'alt_theory' in locals():
        alt_theory = np.clip(alt_theory, -1e12, 1e12)
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Output voltage vs time
    plt.subplot(2, 1, 1)
    plt.plot(time * 1000, vam, 'b-', linewidth=2, label='Vam (Simulated)')  # Convert time to ms
    plt.plot(time * 1000, vam_theory, 'r--', linewidth=2, 
             label=f'Vam (Theory): V₀*e^(t/{1000/growth_rate:.2f}μs)')
    
    if 'alt_theory' in locals() and not np.array_equal(vam_theory, alt_theory):
        plt.plot(time * 1000, alt_theory, 'g-.', linewidth=1.5,
                label=f'Vam (Formula Theory): V₀*e^(t/{1000/theoretical_growth_rate:.2f}μs)')
    
    # Add labels and title
    plt.xlabel('Time [ms]', fontsize=12)
    plt.ylabel('Output Voltage (Vam) [V]', fontsize=12)
    plt.title('AM Circuit Output Voltage vs Time', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    # Plot 2: Op-amp input voltages
    plt.subplot(2, 1, 2)
    plt.plot(time * 1000, op_plus, 'g-', linewidth=2, label='Non-inverting Input (+)')
    plt.plot(time * 1000, op_minus, 'r-', linewidth=2, label='Inverting Input (-)')
    
    # Add labels and title
    plt.xlabel('Time [ms]', fontsize=12)
    plt.ylabel('Input Voltages [V]', fontsize=12)
    plt.title('Op-Amp Input Voltages vs Time', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Ensure directory exists
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
        
    # Save the plot
    plt.savefig('graphs/am_circuit_output.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'graphs/am_circuit_output.png'")
    
    plt.show()
    
    # Generate logarithmic plot to confirm exponential growth
    # Create a copy of the data for log plotting, avoiding negative values
    log_vam = np.abs(vam).copy()  # Use absolute value to handle negative voltages
    log_vam[log_vam < 1e-6] = 1e-6  # Replace very small values
    
    # Create absolute value of theory curve for log plot
    log_vam_theory = np.abs(vam_theory).copy()
    log_vam_theory[log_vam_theory < 1e-6] = 1e-6
    
    plt.figure(figsize=(12, 6))
    plt.semilogy(time * 1000, log_vam, 'b-', linewidth=2, label='|Vam| (Simulated)')
    plt.semilogy(time * 1000, log_vam_theory, 'r--', linewidth=2, 
                 label=f'|Vam| (Theory): |V₀|*e^(t/{1000/growth_rate:.2f}μs)')
    
    if 'alt_theory' in locals() and not np.array_equal(vam_theory, alt_theory):
        log_alt_theory = np.abs(alt_theory).copy()
        log_alt_theory[log_alt_theory < 1e-6] = 1e-6
        plt.semilogy(time * 1000, log_alt_theory, 'g-.', linewidth=1.5,
                    label=f'|Vam| (Formula Theory): |V₀|*e^(t/{1000/theoretical_growth_rate:.2f}μs)')
    
    plt.xlabel('Time [ms]', fontsize=12)
    plt.ylabel('|Output Voltage| (Log Scale) [V]', fontsize=12)
    plt.title('AM Circuit Output Voltage - Logarithmic Scale', fontsize=14)
    plt.grid(True, which='both')
    plt.legend(fontsize=10)
    
    # Save the logarithmic plot
    plt.savefig('graphs/am_circuit_log_output.png', dpi=300, bbox_inches='tight')
    print("Logarithmic plot saved to 'graphs/am_circuit_log_output.png'")
    
    plt.show()
    
    # Analyze and print characteristics of the output
    print("\nCircuit Analysis:")
    print(f"Initial Voltage: {vam[0]:.6f}V")
    print(f"Final Voltage: {vam[-1]:.6f}V")
    print(f"Maximum Voltage: {np.max(vam):.6f}V")
    print(f"Minimum Voltage: {np.min(vam):.6f}V")
    
    # Print growth rates
    print("\nGrowth Rate Analysis:")
    print(f"Fitted Exponential Growth Rate: {growth_rate:.2f} Hz")
    print(f"Time Constant from Fit: {1000/growth_rate:.3f} μs")

if __name__ == '__main__':
    # Set up logging
    logger = Logging.setup_logging()
    
    # Run the main simulation
    simulate_am_circuit()