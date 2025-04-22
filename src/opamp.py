from PySpice.Spice.Netlist import SubCircuitFactory
from PySpice.Unit import u_kΩ, u_uF, u_V, u_Hz, u_MHz, u_MΩ, u_Ω, u_us, u_ms, kilo, u_mV


class BasicOperationalAmplifier(SubCircuitFactory):

    NAME = 'BasicOperationalAmplifier'
    NODES = ('non_inverting_input', 'inverting_input', 'output', 'vdd', 'vss')

    def __init__(self):

        super().__init__()

        # Input impedance
        self.R('input', 'non_inverting_input', 'inverting_input', 10@u_MΩ)

        # dc gain=100k and pole1=100hz
        # unity gain = dcgain x pole1 = 10MHZ
        self.VCVS('gain', 1, self.gnd, 'non_inverting_input', 'inverting_input', voltage_gain=kilo(100))
        self.R('P1', 1, 2, 1@u_kΩ)
        self.C('P1', 2, self.gnd, 1.5915@u_uF)

        # Output buffer and resistance
        self.VCVS('buffer', 3, self.gnd, 2, self.gnd, 1)
        self.R('out', 3, 4, 10@u_Ω)
        
        # Realistic output limiting using diodes to clip output voltage
        # This creates a voltage limiter that restricts output to within power supply rails
        self.D('upper', 4, 'vdd', model='Dlimit')
        self.D('lower', 'vss', 4, model='Dlimit')
        
        # Output connection
        self.R('output_r', 4, 'output', 1@u_Ω)
        
        # Define diode model with low resistance when forward biased
        self.model('Dlimit', 'D', is_=1e-14, rs=1)


# Code to test the Op-Amp
if __name__ == '__main__':
    import PySpice.Logging.Logging as Logging
    from PySpice.Spice.Netlist import Circuit
    import matplotlib.pyplot as plt
    import numpy as np
    
    logger = Logging.setup_logging()
    
    # Create a circuit
    circuit = Circuit('Op-Amp Test')
    
    # Add the Op-Amp subcircuit
    circuit.subcircuit(BasicOperationalAmplifier())
    
    # Add power supplies
    Vdd = 5   # Positive supply
    Vss = 0   # Ground/negative supply (could be negative for dual supply)
    circuit.V('dd', 'vdd', circuit.gnd, Vdd@u_V)
    circuit.V('ss', 'vss', circuit.gnd, Vss@u_V)
    
    # Instantiate the Op-Amp with power supplies
    circuit.X('opamp1', 'BasicOperationalAmplifier', 'in_plus', 'in_minus', 'out', 'vdd', 'vss')
    
    # Add input voltage sources - store references to modify later
    v_in_plus = circuit.V('in_plus', 'in_plus', circuit.gnd, 0@u_V)
    v_in_minus = circuit.V('in_minus', 'in_minus', circuit.gnd, 0@u_V)
    
    # Create simulator
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    
    # Test 1: DC sweep of non-inverting input (differential mode)
    print("Simulating differential mode operation")
    analysis1 = simulator.dc(Vin_plus=slice(-0.1, 0.1, 0.001))
    
    # Test 2: Saturation behavior - sweep input beyond rails
    print("Simulating saturation behavior")
    analysis2 = simulator.dc(Vin_plus=slice(-1, 1, 0.01))
    
    # Test 3: Frequency response (AC analysis)
    print("Simulating frequency response")
    v_in_plus.dc_value = 0@u_V  # DC bias point
    v_in_plus.ac_value = 1@u_mV  # AC small signal amplitude
    v_in_minus.dc_value = 0@u_V
    analysis3 = simulator.ac(start_frequency=1@u_Hz, stop_frequency=100@u_MHz, number_of_points=1000, variation='dec')
    
    # For transient analysis we need a new circuit since we can't change the source type
    print("Simulating step response")
    transient_circuit = Circuit('Op-Amp Transient Test')
    transient_circuit.subcircuit(BasicOperationalAmplifier())
    
    # Add power supplies to transient circuit
    transient_circuit.V('dd', 'vdd', transient_circuit.gnd, Vdd@u_V)
    transient_circuit.V('ss', 'vss', transient_circuit.gnd, Vss@u_V)
    
    # Instantiate the Op-Amp with power supplies
    transient_circuit.X('opamp1', 'BasicOperationalAmplifier', 'in_plus', 'in_minus', 'out', 'vdd', 'vss')
    
    # Define a pulse input for transient analysis
    transient_circuit.PulseVoltageSource('in_plus', 'in_plus', transient_circuit.gnd, 
                                         initial_value=0@u_V, pulsed_value=10@u_mV, 
                                         delay_time=0.1@u_ms, rise_time=10@u_us, fall_time=10@u_us, 
                                         pulse_width=1@u_ms, period=2@u_ms)
    transient_circuit.V('in_minus', 'in_minus', transient_circuit.gnd, 0@u_V)
    
    # Run transient simulation
    transient_simulator = transient_circuit.simulator(temperature=25, nominal_temperature=25)
    analysis4 = transient_simulator.transient(step_time=10@u_us, end_time=3@u_ms)
    
    # Plot the results
    figure, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot for DC sweep (differential mode)
    ax1 = axes[0, 0]
    ax1.set_title('Op-Amp DC Response (Small-Signal)')
    ax1.set_xlabel('Differential Input Voltage [V]')
    ax1.set_ylabel('Output Voltage [V]')
    ax1.grid(True)
    ax1.plot(analysis1['v-sweep'], analysis1['out'])
    
    # Plot for DC sweep (saturation behavior)
    ax2 = axes[0, 1]
    ax2.set_title('Op-Amp Saturation Behavior')
    ax2.set_xlabel('Input Voltage [V]')
    ax2.set_ylabel('Output Voltage [V]')
    ax2.grid(True)
    ax2.plot(analysis2['v-sweep'], analysis2['out'])
    
    # Add horizontal lines showing supply rails
    ax2.axhline(y=Vdd, color='r', linestyle='--', alpha=0.7, label=f'Vdd = {Vdd}V')
    ax2.axhline(y=Vss, color='b', linestyle='--', alpha=0.7, label=f'Vss = {Vss}V')
    ax2.legend()
    
    # Plot for AC analysis - Magnitude response
    ax3 = axes[1, 0]
    ax3.set_title('Op-Amp Frequency Response')
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('Gain [dB]')
    ax3.grid(True)
    ax3.semilogx(analysis3.frequency, 20*np.log10(np.abs(analysis3.out)))
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # 0dB line
    
    # Plot for transient analysis
    ax4 = axes[1, 1]
    ax4.set_title('Op-Amp Step Response')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Voltage [V]')
    ax4.grid(True)
    ax4.plot(analysis4.time, analysis4['in_plus'], label='Input')
    ax4.plot(analysis4.time, analysis4['out'], label='Output')
    ax4.legend()
    
    plt.tight_layout()
    
    # Create a directory for graphs if it doesn't exist
    import os
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    
    # Save the plot to a PNG file
    plt.savefig('graphs/opamp_simulation_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'graphs/opamp_simulation_results.png'")
    
    plt.show()
    
    # Print some key characteristics
    dc_gain = float(analysis1['out'].max() - analysis1['out'].min()) / 0.2  # 0.2V is the input swing
    print(f"\nApproximate DC Gain: {dc_gain:.2f}")
    
    # Find unity gain frequency (0dB crossing point)
    gain_db = 20*np.log10(np.abs(analysis3.out))
    freq = analysis3.frequency
    for i in range(len(gain_db)-1):
        if gain_db[i] > 0 and gain_db[i+1] < 0:
            unity_gain_freq = freq[i]
            break
    else:
        unity_gain_freq = "Not found in simulation range"
    
    print(f"Unity Gain Frequency: {unity_gain_freq}")
    
    # Print saturation voltages
    sat_high = float(analysis2['out'].max())
    sat_low = float(analysis2['out'].min())
    print(f"Positive Saturation Voltage: {sat_high:.3f}V")
    print(f"Negative Saturation Voltage: {sat_low:.3f}V")