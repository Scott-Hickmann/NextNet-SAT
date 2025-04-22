import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
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

def create_am_circuit():
    """
    Create an AM circuit with an opamp, two capacitors of capacitance C,
    and two resistors of resistance R, configured as a positive feedback oscillator.
    
    This configuration will produce exponential growth in the output voltage.
    """
    # Create a circuit
    circuit = Circuit('AM Circuit')
    
    # Add the opamp subcircuit
    circuit.subcircuit(BasicOperationalAmplifier())
    
    # Define component values - carefully tuned for proper oscillation
    R = 100e3@u_Ω     # Feedback resistor
    C = 0.01e-6@u_F     # 0.01 μF capacitors - smaller for faster response
    
    # Add power supplies
    Vdd = 1.0    # Increased supply voltage for wider swing
    Vss = -1.0   # Negative supply to allow for negative swing
    circuit.V('dd', 'vdd', circuit.gnd, Vdd@u_V)
    circuit.V('ss', 'vss', circuit.gnd, Vss@u_V)
    
    # Define nodes
    op_plus = 'op_plus'    # Non-inverting input of opamp
    op_minus = 'op_minus'  # Inverting input of opamp
    vam = 'vam'            # Output of opamp
    
    circuit.R('1', vam, op_plus, R)
    circuit.R('2', op_minus, circuit.gnd, R)

    circuit.C('1', op_minus, circuit.gnd, C)
    circuit.C('2', op_plus, circuit.gnd, C)
    
    # Instantiate the opamp
    circuit.X('opamp', 'BasicOperationalAmplifier', 
              op_plus, op_minus, vam, 'vdd', 'vss')
    
    # Add initial condition to kickstart oscillation
    circuit.PulseVoltageSource('kick', 'kick', circuit.gnd,
                               initial_value=0@u_V,
                               pulsed_value=0.1@u_V,  # Stronger pulse
                               pulse_width=10@u_us,
                               period=1@u_ms,
                               delay_time=0@u_us,
                               rise_time=1@u_us,
                               fall_time=1@u_us)
    
    # Connect the kick source directly to the non-inverting input
    circuit.R('kick', 'kick', op_plus, 1@u_kΩ)  # Lower resistance for stronger effect
    
    return circuit, R.value, C.value

def simulate_am_circuit():
    """
    Simulate the AM circuit and plot the output.
    """
    # Create the circuit and get component values
    circuit, R, C = create_am_circuit()
    
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
    op_plus = np.array(analysis['op_plus'])
    op_minus = np.array(analysis['op_minus'])
    
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
    # print(f"Theoretical Aeff: {Aeff}")

    
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