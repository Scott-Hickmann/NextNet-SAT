import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import u_V, u_kΩ, u_Ω, u_ms, u_us, u_F
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
    
    Each pair of variable resistances shares the same control signals (vi, vi_bar, q, q_bar).
    
    Nodes:
    - vout: Output voltage of the circuit
    - vdd: Positive power supply
    - vss: Negative power supply
    - gnd: Ground reference
    - vi1: Control voltage 1 for first pair of variable resistances
    - vi_bar1: Complement of vi1
    - q1: Control signal 1 for first pair of variable resistances
    - q_bar1: Complement of q1
    - vi2: Control voltage 2 for second pair of variable resistances
    - vi_bar2: Complement of vi2
    - q2: Control signal 2 for second pair of variable resistances
    - q_bar2: Complement of q2
    - vi3: Control voltage 3 for third pair of variable resistances
    - vi_bar3: Complement of vi3
    - q3: Control signal 3 for third pair of variable resistances
    - q_bar3: Complement of q3
    - op_plus: Non-inverting input of the op-amp (exposed for measurement)
    - op_minus: Inverting input of the op-amp (exposed for measurement)
    """
    NODES = ('vout', 'vdd', 'vss', 'gnd', 
             'vi1', 'vi_bar1', 'q1', 'q_bar1',
             'vi2', 'vi_bar2', 'q2', 'q_bar2', 
             'vi3', 'vi_bar3', 'q3', 'q_bar3',
             'op_plus', 'op_minus')
    
    def __init__(self, C=0.01e-6):
        """
        Initialize the AMFull subcircuit with specified capacitance.
        
        Args:
            C (float): Capacitance value in farads (default: 0.01μF)
        """
        super().__init__(f'AMFull_{C}', *self.NODES)
            
        # Add the AMCircuit subcircuit
        am = AMCircuit(C=C)
        self.subcircuit(am)
        
        # Add the VariableResistance subcircuit
        var_res = VariableResistance()
        self.subcircuit(var_res)
        
        # Define internal nodes for connecting resistances in series
        # For R1 (first group of 3 variable resistances)
        r1_node1 = 'r1_node1'  # Between first and second resistance
        r1_node2 = 'r1_node2'  # Between second and third resistance
        
        # For R2 (second group of 3 variable resistances)
        r2_node1 = 'r2_node1'  # Between first and second resistance
        r2_node2 = 'r2_node2'  # Between second and third resistance
        
        # Instantiate the AMCircuit
        self.X('am', am.name, 'vout', 'r1_a', 'r1_b', 'r2_a', 'r2_b', 
               'vdd', 'vss', 'gnd', 'op_plus', 'op_minus')
        
        # Instantiate the 6 variable resistances
        
        # First group (R1) - 3 variable resistances in series
        # First pair - Variable Resistance 1.1
        self.X('var_res11', var_res.NAME, 'vi1', 'vi_bar1', 'q1', 'q_bar1', 
               'r1_a', r1_node1, 'vdd', 'gnd')
        
        # Second pair - Variable Resistance 1.2
        self.X('var_res12', var_res.NAME, 'vi2', 'vi_bar2', 'q2', 'q_bar2', 
               r1_node1, r1_node2, 'vdd', 'gnd')
        
        # Third pair - Variable Resistance 1.3
        self.X('var_res13', var_res.NAME, 'vi3', 'vi_bar3', 'q3', 'q_bar3', 
               r1_node2, 'r1_b', 'vdd', 'gnd')
        
        # Second group (R2) - 3 variable resistances in series
        # First pair - Variable Resistance 2.1
        self.X('var_res21', var_res.NAME, 'vi1', 'vi_bar1', 'q1', 'q_bar1', 
               'r2_a', r2_node1, 'vdd', 'gnd')
        
        # Second pair - Variable Resistance 2.2
        self.X('var_res22', var_res.NAME, 'vi2', 'vi_bar2', 'q2', 'q_bar2', 
               r2_node1, r2_node2, 'vdd', 'gnd')
        
        # Third pair - Variable Resistance 2.3
        self.X('var_res23', var_res.NAME, 'vi3', 'vi_bar3', 'q3', 'q_bar3', 
               r2_node2, 'r2_b', 'vdd', 'gnd')

def test_am_full_circuit():
    """
    Test the AMFull circuit under two scenarios:
    1. q1 = q2 = q3 = GND (all q values are 0)
    2. q1 = q2 = q3 = VDD (all q values are 1)
    
    For each scenario, vi1 = vi2 = vi3 is ramped from GND to VDD.
    """
    # Create a circuit
    circuit = Circuit('AMFull Test Circuit')
    
    # Define component values
    C = 0.01e-6  # 0.01µF
    
    # Add power supplies
    Vdd = 1.0
    Vss = -1.0
    circuit.V('dd', 'vdd', circuit.gnd, Vdd@u_V)
    circuit.V('ss', 'vss', circuit.gnd, Vss@u_V)
    
    # Create the AMFull subcircuit
    am_full = AMFull(C=C)
    circuit.subcircuit(am_full)
    
    # Function to run simulation for a specific scenario
    def simulate_scenario(scenario_name, q_value):
        print(f"\nSimulating {scenario_name}...")
        
        # Create a new circuit for this scenario
        scenario_circuit = Circuit(f'AMFull Test - {scenario_name}')
        
        # Add power supplies
        scenario_circuit.V('dd', 'vdd', scenario_circuit.gnd, Vdd@u_V)
        scenario_circuit.V('ss', 'vss', scenario_circuit.gnd, Vss@u_V)
        
        # Add stabilizing capacitors to power rails (helps with convergence)
        scenario_circuit.C('stab_vdd', 'vdd', scenario_circuit.gnd, 1e-9@u_F)
        scenario_circuit.C('stab_vss', 'vss', scenario_circuit.gnd, 1e-9@u_F)
        
        # Add the AMFull subcircuit
        scenario_circuit.subcircuit(am_full)
        
        # Instantiate the AMFull circuit
        scenario_circuit.X('am_full', am_full.name, 'vout', 'vdd', 'vss', scenario_circuit.gnd,
                          'vi', 'vi_bar', 'q1', 'q_bar1',
                          'vi', 'vi_bar', 'q2', 'q_bar2',
                          'vi', 'vi_bar', 'q3', 'q_bar3',
                          'op_plus', 'op_minus')
        
        # Set q values based on the scenario
        q_voltage = q_value@u_V  # 0 for scenario 1, 1 for scenario 2
        q_bar_voltage = (1 - q_value)@u_V  # Complement
        
        # Add control signals with fixed values for q and q_bar
        # Add small resistors in series to improve convergence
        scenario_circuit.V('q1_src', 'q1_src', scenario_circuit.gnd, q_voltage)
        scenario_circuit.R('q1_r', 'q1_src', 'q1', 10@u_Ω)
        
        scenario_circuit.V('q_bar1_src', 'q_bar1_src', scenario_circuit.gnd, q_bar_voltage)
        scenario_circuit.R('q_bar1_r', 'q_bar1_src', 'q_bar1', 10@u_Ω)
        
        scenario_circuit.V('q2_src', 'q2_src', scenario_circuit.gnd, q_voltage)
        scenario_circuit.R('q2_r', 'q2_src', 'q2', 10@u_Ω)
        
        scenario_circuit.V('q_bar2_src', 'q_bar2_src', scenario_circuit.gnd, q_bar_voltage)
        scenario_circuit.R('q_bar2_r', 'q_bar2_src', 'q_bar2', 10@u_Ω)
        
        scenario_circuit.V('q3_src', 'q3_src', scenario_circuit.gnd, q_voltage)
        scenario_circuit.R('q3_r', 'q3_src', 'q3', 10@u_Ω)
        
        scenario_circuit.V('q_bar3_src', 'q_bar3_src', scenario_circuit.gnd, q_bar_voltage)
        scenario_circuit.R('q_bar3_r', 'q_bar3_src', 'q_bar3', 10@u_Ω)
        
        # Use a more stable approach for ramping vi and vi_bar with PWL (piece-wise linear)
        # Define time points for the PWL sources
        end_time_value = float(0.5e-3)  # 0.5ms in seconds
        time_points = [0, end_time_value/2, end_time_value]
        
        # Define vi voltage points (ramp from 0 to VDD)
        vi_points = [0, Vdd/2, Vdd]
        # Define vi_bar voltage points (complement of vi)
        vi_bar_points = [Vdd, Vdd/2, 0]
        
        # Create PWL (Piece-Wise Linear) sources for vi and vi_bar
        scenario_circuit.PieceWiseLinearVoltageSource('vi_src', 'vi_src', scenario_circuit.gnd,
                                                  values=[(t, v) for t, v in zip(time_points, vi_points)])
        scenario_circuit.R('vi_r', 'vi_src', 'vi', 10@u_Ω)
        
        scenario_circuit.PieceWiseLinearVoltageSource('vi_bar_src', 'vi_bar_src', scenario_circuit.gnd,
                                                  values=[(t, v) for t, v in zip(time_points, vi_bar_points)])
        scenario_circuit.R('vi_bar_r', 'vi_bar_src', 'vi_bar', 10@u_Ω)
        
        # Add initial condition to kickstart oscillation
        scenario_circuit.PulseVoltageSource('kick', 'kick', scenario_circuit.gnd,
                                         initial_value=0@u_V,
                                         pulsed_value=0.1@u_V,
                                         pulse_width=10@u_us,
                                         period=1@u_ms,
                                         delay_time=0@u_us,
                                         rise_time=1@u_us,
                                         fall_time=1@u_us)
        
        # Connect the kick source to the op_plus node
        scenario_circuit.R('kick', 'kick', 'op_plus', 1@u_kΩ)
        
        # Create simulator with improved options for convergence
        simulator = scenario_circuit.simulator(temperature=25, nominal_temperature=25)
        
        # Set simulation parameters - shorter time to see exponential growth
        step_time = 0.5@u_us  # Increased step time for better convergence
        end_time = 0.5@u_ms   # Same as in am.py
        
        # Set analysis parameters to improve convergence
        simulator.options(reltol=1e-2,  # Relaxed relative tolerance
                         abstol=1e-5,   # Relaxed absolute tolerance
                         method='gear',
                         gmin=1e-10,    # Minimum conductance
                         maxord=2,      # Maximum order for integration method
                         itl1=500,      # Increase DC iteration limit
                         itl4=500)      # Increase transient iteration limit
        
        print(f"Running transient simulation for scenario {scenario_name}...")
        analysis = simulator.transient(step_time=step_time, end_time=end_time)
        return analysis
    
    # Run both scenarios with better error handling
    try:
        # Try to run both scenarios
        analysis1 = simulate_scenario("Scenario 1 (q=GND)", 0)
        analysis2 = simulate_scenario("Scenario 2 (q=VDD)", 1)
        
        # Process and plot results
        time1 = np.array(analysis1.time)
        vout1 = np.array(analysis1['vout'])
        vi1 = np.array(analysis1['vi'])
        vi_bar1 = np.array(analysis1['vi_bar'])
        op_plus1 = np.array(analysis1['op_plus'])
        op_minus1 = np.array(analysis1['op_minus'])
        
        time2 = np.array(analysis2.time)
        vout2 = np.array(analysis2['vout'])
        vi2 = np.array(analysis2['vi'])
        vi_bar2 = np.array(analysis2['vi_bar'])
        op_plus2 = np.array(analysis2['op_plus'])
        op_minus2 = np.array(analysis2['op_minus'])
        
        # Create plots
        plt.figure(figsize=(15, 15))
        
        # Plot for Scenario 1
        plt.subplot(3, 2, 1)
        plt.plot(time1 * 1000, vout1, 'b-', linewidth=2, label='Vout')
        plt.plot(time1 * 1000, vi1, 'r-', linewidth=1, label='Vi')
        plt.plot(time1 * 1000, vi_bar1, 'g-', linewidth=1, label='Vi_bar')
        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage [V]')
        plt.title('Scenario 1 (q=GND) - Output and Control Voltages')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 2, 2)
        plt.plot(time1 * 1000, op_plus1, 'g-', linewidth=2, label='Op+')
        plt.plot(time1 * 1000, op_minus1, 'r-', linewidth=2, label='Op-')
        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage [V]')
        plt.title('Scenario 1 (q=GND) - Op-Amp Input Voltages')
        plt.grid(True)
        plt.legend()
        
        # Plot for Scenario 2
        plt.subplot(3, 2, 3)
        plt.plot(time2 * 1000, vout2, 'b-', linewidth=2, label='Vout')
        plt.plot(time2 * 1000, vi2, 'r-', linewidth=1, label='Vi')
        plt.plot(time2 * 1000, vi_bar2, 'g-', linewidth=1, label='Vi_bar')
        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage [V]')
        plt.title('Scenario 2 (q=VDD) - Output and Control Voltages')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 2, 4)
        plt.plot(time2 * 1000, op_plus2, 'g-', linewidth=2, label='Op+')
        plt.plot(time2 * 1000, op_minus2, 'r-', linewidth=2, label='Op-')
        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage [V]')
        plt.title('Scenario 2 (q=VDD) - Op-Amp Input Voltages')
        plt.grid(True)
        plt.legend()
        
        # Comparison plot - Both scenarios
        plt.subplot(3, 1, 3)
        plt.plot(time1 * 1000, vout1, 'b-', linewidth=2, label='Vout (q=GND)')
        plt.plot(time2 * 1000, vout2, 'r-', linewidth=2, label='Vout (q=VDD)')
        plt.plot(time1 * 1000, vi1, 'g--', linewidth=1, label='Vi (ramping)')
        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage [V]')
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
        log_vout1 = np.abs(vout1)
        log_vout1[log_vout1 < 1e-6] = 1e-6  # Replace very small values
        
        log_vout2 = np.abs(vout2)
        log_vout2[log_vout2 < 1e-6] = 1e-6  # Replace very small values
        
        plt.semilogy(time1 * 1000, log_vout1, 'b-', linewidth=2, label='|Vout| (q=GND)')
        plt.semilogy(time2 * 1000, log_vout2, 'r-', linewidth=2, label='|Vout| (q=VDD)')
        
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
        for name, t, vout in [("Scenario 1 (q=GND)", time1, vout1), 
                              ("Scenario 2 (q=VDD)", time2, vout2)]:
            if np.all(vout == 0):
                print(f"\n{name} Analysis: Simulation failed, no valid data to analyze")
                continue
                
            print(f"\n{name} Analysis:")
            print(f"Initial Voltage: {vout[0]:.6f}V")
            print(f"Final Voltage: {vout[-1]:.6f}V")
            print(f"Maximum Voltage: {np.max(vout):.6f}V")
            print(f"Minimum Voltage: {np.min(vout):.6f}V")
            
            # Try to estimate the growth rate using log-linear regression
            # First, find a section with clear exponential growth
            log_vout = np.log(np.abs(vout))
            log_vout[~np.isfinite(log_vout)] = -20  # Replace non-finite values
            
            # Take a section that shows exponential behavior
            start_idx = len(vout) // 4  # Skip the initial part
            end_idx = 3 * len(vout) // 4  # Skip the saturation part
            
            if start_idx < end_idx:
                time_section = t[start_idx:end_idx]
                log_vout_section = log_vout[start_idx:end_idx]
                
                try:
                    from scipy.stats import linregress
                    slope, intercept, r_value, _, _ = linregress(time_section, log_vout_section)
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