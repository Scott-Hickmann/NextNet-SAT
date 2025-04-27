import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import u_V, u_Ω, u_ms, u_us, u_F
import matplotlib.pyplot as plt
import numpy as np
import os

from am import AMCircuit
from variable_resistance import VariableResistance

class AMFull(SubCircuit):
    """
    AMFull subcircuit that combines an AMCircuit with 6 VariableResistance subcircuits.
    
    The 6 variable resistances are split into two groups of 3, with each group connected in series:
    - The first group forms the first resistor (R1) for the AM circuit
    - The second group forms the second resistor (R2) for the AM circuit
    
    Each variable resistance only needs vi and cmi (-1 or 1) parameters.
    
    Nodes:
    - vam: Output voltage of the circuit
    - gnd: Ground reference
    - vi1: Control voltage 1 for first pair of variable resistances
    - vi2: Control voltage 2 for second pair of variable resistances
    - vi3: Control voltage 3 for third pair of variable resistances
    """
    NODES = ('vam', 'n1', 'gnd', 'vi1', 'vi2', 'vi3')
    
    def __init__(self, cm1, cm2, cm3, C=0.01e-6, gain=2.0):
        """
        Initialize the AMFull subcircuit with specified capacitance.
        
        Args:
            cm1: cmi value for the first variable resistance
            cm2: cmi value for the second variable resistance
            cm3: cmi value for the third variable resistance
            C (float): Capacitance value in farads (default: 0.01μF)
            gain (float): Gain of the AM circuit (default: 2.0)
        """
        super().__init__(f'AMFull_{C}', *self.NODES)
            
        # Add the AMCircuit subcircuit
        am = AMCircuit(C=C, gain=gain)
        self.subcircuit(am)
        
        # Add the VariableResistance subcircuits
        var_res1 = VariableResistance(cmi=cm1)
        var_res2 = VariableResistance(cmi=cm2)
        var_res3 = VariableResistance(cmi=cm3)
        
        self.subcircuit(var_res1)
        self.subcircuit(var_res2)
        self.subcircuit(var_res3)
        
        # Define internal nodes for connecting resistances in series
        r_node1 = 'r1_node1'  # Between first and second resistance
        r_node2 = 'r1_node2'  # Between second and third resistance
        
        # Instantiate the AMCircuit
        self.X('am1', am.name, 'vam', 'n1', 'gnd')
        
        # Instantiate the 3 variable resistances
        DEBUG = False
        if DEBUG:
            R = 0.3e3@u_Ω
            self.R('r11', am.RA, r_node1, R)
            self.R('r12', r_node1, r_node2, R)
            self.R('r13', r_node2, am.RB, R)
        else:
            # 3 variable resistances in series
            self.X('var_res11', var_res1.NAME, 'vi1', am.RA, r_node1, 'vdd', 'gnd')
            self.X('var_res12', var_res2.NAME, 'vi2', r_node1, r_node2, 'vdd', 'gnd')
            self.X('var_res13', var_res3.NAME, 'vi3', r_node2, am.RB, 'vdd', 'gnd')


# Function to run simulation for a specific scenario
def simulate_scenario(scenario_name, cmi_values, vi_points):
    # Define component values
    C = 0.01e-6  # 0.01µF
    gain = 2.0

    print(f"\nSimulating {scenario_name}...")
    
    # Create a new circuit for this scenario
    scenario_circuit = Circuit(f'AMFull Test - {scenario_name}')
    
    # Create the AMFull subcircuit with the specified cmi values
    am_full = AMFull(cm1=cmi_values[0], cm2=cmi_values[1], cm3=cmi_values[2], C=C, gain=gain)
    scenario_circuit.subcircuit(am_full)
    
    # Instantiate the AMFull circuit
    scenario_circuit.X('am_full', am_full.name, 'vam', 'n1', scenario_circuit.gnd, 'vi', 'vi', 'vi')
    
    # Use a more stable approach for ramping vi with PWL (piece-wise linear)
    # Define time points for the PWL source
    end_time_value = float(0.5e-3)  # 0.5ms in seconds
    time_points = [0, end_time_value/2, end_time_value]
    
    # Create PWL (Piece-Wise Linear) source for vi
    scenario_circuit.PieceWiseLinearVoltageSource('vi_src', 'vi_src', scenario_circuit.gnd,
                                                values=[(t, v) for t, v in zip(time_points, vi_points)])
    scenario_circuit.R('vi_r', 'vi_src', 'vi', 10@u_Ω)

    # Set simulation parameters - shorter time to see exponential growth
    step_time = 0.1@u_us  # Increased step time for better convergence
    end_time = 0.5@u_ms   # Same as in am.py
    
    # Create simulator with improved options for convergence
    simulator = scenario_circuit.simulator(temperature=25, nominal_temperature=25)

    # Set initial condition on n1 node 
    simulator.initial_condition(n1=0.01)
    
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

    print(f"Running transient simulation for scenario {scenario_name}...")
    analysis = simulator.transient(step_time=step_time, end_time=end_time)
    return analysis

def test_am_full_circuit():
    """
    Test the AMFull circuit under two scenarios:
    1. All cmi values = -1
    2. All cmi values = 1
    
    For each scenario, vi1 = vi2 = vi3 is ramped from GND to VDD.
    """
    
    # Define cmi configurations for each scenario
    scenario1_cmis = [-1, -1, -1]  # All -1
    scenario2_cmis = [1, 1, 1] # All 1

    Vdd = 1.0
    # Unsatisfied
    # scenario1_vi_points = [Vdd, Vdd]
    # scenario2_vi_points = [0, 0]
    # Satisfied
    # scenario1_vi_points = [0, 0]
    # scenario2_vi_points = [Vdd, Vdd]
    # Towards satisfied
    scenario1_vi_points = [Vdd/2, 0, 0]
    scenario2_vi_points = [Vdd/2, Vdd, Vdd]

    # Run both scenarios with better error handling
    try:
        # Try to run both scenarios
        analysis1 = simulate_scenario("Scenario 1 (cmi=-1)", scenario1_cmis, scenario1_vi_points)
        analysis2 = simulate_scenario("Scenario 2 (cmi=1)", scenario2_cmis, scenario2_vi_points)
        
        # Process and plot results
        time1 = np.array(analysis1.time)
        vam1 = np.array(analysis1['vam'])
        vi1 = np.array(analysis1['vi'])
        
        time2 = np.array(analysis2.time)
        vam2 = np.array(analysis2['vam'])
        vi2 = np.array(analysis2['vi'])
        
        # Create plots
        plt.figure(figsize=(15, 15))

        EPS = 0.01
        
        # Plot for Scenario 1
        plt.subplot(2, 2, 1)
        plt.plot(time1 * 1000, vam1, 'b-', linewidth=2, label='Vam')
        plt.plot(time1 * 1000, vi1, 'b--', linewidth=1, label='Vi')
        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage [V]')
        plt.ylim(-EPS, Vdd+EPS)
        plt.title('Scenario 1 (cmi=-1) - Output and Control Voltages')
        plt.grid(True)
        plt.legend()
        
        # Plot for Scenario 2
        plt.subplot(2, 2, 2)
        plt.plot(time2 * 1000, vam2, 'r-', linewidth=2, label='Vam')
        plt.plot(time2 * 1000, vi2, 'r--', linewidth=1, label='Vi')
        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage [V]')
        plt.ylim(-EPS, Vdd+EPS)
        plt.title('Scenario 2 (cmi=1) - Output and Control Voltages')
        plt.grid(True)
        plt.legend()
        
        # Comparison plot - Both scenarios
        plt.subplot(2, 1, 2)
        plt.plot(time1 * 1000, vam1, 'b-', linewidth=2, label='Vam (cmi=-1)')
        plt.plot(time1 * 1000, vi1, 'b--', linewidth=1, label='Vi 1')
        plt.plot(time2 * 1000, vam2, 'r-', linewidth=2, label='Vam (cmi=1)')
        plt.plot(time2 * 1000, vi2, 'r--', linewidth=1, label='Vi 2')
        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage [V]')
        plt.ylim(-EPS, Vdd+EPS)
        plt.title('Comparison of Both Scenarios - Output Voltage')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Ensure directory exists
        if not os.path.exists('graphs'):
            os.makedirs('graphs')
            
        # Save the plot
        plt.savefig('graphs/am_full_results.png', dpi=300, bbox_inches='tight')
        print("Plot saved to 'graphs/am_full_results.png'")
        
        plt.show()
        
        # Generate logarithmic plots to compare growth rates
        plt.figure(figsize=(12, 6))
        
        # Prepare log data for both scenarios
        log_vam1 = np.abs(vam1)
        log_vam1[log_vam1 < 1e-6] = 1e-6  # Replace very small values
        
        log_vam2 = np.abs(vam2)
        log_vam2[log_vam2 < 1e-6] = 1e-6  # Replace very small values
        
        plt.semilogy(time1 * 1000, log_vam1, 'b-', linewidth=2, label='|Vam| (cmi=-1)')
        plt.semilogy(time2 * 1000, log_vam2, 'r-', linewidth=2, label='|Vam| (cmi=1)')
        
        plt.xlabel('Time [ms]', fontsize=12)
        plt.ylabel('|Output Voltage| (Log Scale) [V]', fontsize=12)
        plt.title('AMFull Circuit Output Voltage - Logarithmic Scale', fontsize=14)
        plt.grid(True, which='both')
        plt.legend(fontsize=10)
        
        # Save the logarithmic plot
        plt.savefig('graphs/am_full_log_output.png', dpi=300, bbox_inches='tight')
        print("Logarithmic plot saved to 'graphs/am_full_log_output.png'")
        
        plt.show()
        
        # Print analysis for both scenarios
        for name, t, vam in [("Scenario 1 (cmi=-1)", time1, vam1), 
                              ("Scenario 2 (cmi=1)", time2, vam2)]:
            if np.all(vam == 0):
                print(f"\n{name} Analysis: Simulation failed, no valid data to analyze")
                continue
                
            print(f"\n{name} Analysis:")
            print(f"Initial Voltage: {vam[0]:.6f}V")
            print(f"Final Voltage: {vam[-1]:.6f}V")
            print(f"Maximum Voltage: {np.max(vam):.6f}V")
            print(f"Minimum Voltage: {np.min(vam):.6f}V")
            
            # Try to estimate the growth rate using log-linear regression
            # First, find a section with clear exponential growth
            log_vam = np.log(np.abs(vam))
            log_vam[~np.isfinite(log_vam)] = -20  # Replace non-finite values
            
            # Take a section that shows exponential behavior
            start_idx = len(vam) // 4  # Skip the initial part
            end_idx = 3 * len(vam) // 4  # Skip the saturation part
            
            if start_idx < end_idx:
                time_section = t[start_idx:end_idx]
                log_vam_section = log_vam[start_idx:end_idx]
                
                try:
                    from scipy.stats import linregress
                    slope, intercept, r_value, _, _ = linregress(time_section, log_vam_section)
                    growth_rate = slope
                    print(f"Estimated Growth Rate: {growth_rate:.2f} Hz")
                    print(f"Time Constant: {1000/growth_rate:.3f} μs")
                    print(f"R-squared of fit: {r_value**2:.6f}")
                except Exception as e:
                    print(f"Could not estimate growth rate: {str(e)}")
    
    except Exception as e:
        print(f"Error during simulation or analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Set up logging
    logger = Logging.setup_logging()
    
    # Run the test function
    test_am_full_circuit() 