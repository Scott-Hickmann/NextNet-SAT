import PySpice.Logging.Logging as Logging

from PySpice.Spice.Netlist import Circuit, SubCircuitFactory
from PySpice.Unit import u_V
import matplotlib.pyplot as plt
import numpy as np


class NorGate(SubCircuitFactory):
    NAME = 'nor_gate'
    NODES = ('input_a', 'input_b', 'output', 'vdd', 'gnd')
    
    def __init__(self):
        super().__init__()
        
        # Define the NMOS and PMOS models with parameters
        self.model('NMOS', 'NMOS', vto=0, lambda_=1)
        self.model('PMOS', 'PMOS', vto=0, lambda_=1)
        
        # PMOS transistors in series (from VDD to output)
        # M <name> <drain node> <gate node> <source node> <bulk/substrate node>
        self.M(1, 'node1', 'input_a', 'vdd', 'vdd', model='PMOS')
        self.M(2, 'output', 'input_b', 'node1', 'vdd', model='PMOS')
        
        # NMOS transistors in parallel (from output to GND)
        self.M(3, 'output', 'input_a', 'gnd', 'gnd', model='NMOS')
        self.M(4, 'output', 'input_b', 'gnd', 'gnd', model='NMOS')


# Test plots for the NOR gate
if __name__ == '__main__':
    # Set up logging
    logger = Logging.setup_logging()
    
    # Create a circuit
    circuit = Circuit('NOR Gate Test')
    
    # Add the NOR gate subcircuit
    circuit.subcircuit(NorGate())
    
    # Instantiate the NOR gate
    circuit.X('nor1', 'nor_gate', 'in_a', 'in_b', 'out', 'vdd', circuit.gnd)
    
    # Add power supply
    vdd = 1
    circuit.V('dd', 'vdd', circuit.gnd, vdd@u_V)
    
    # Add input voltage sources - store references to them
    v_in_a = circuit.V('in_a', 'in_a', circuit.gnd, 0@u_V)
    v_in_b = circuit.V('in_b', 'in_b', circuit.gnd, 0@u_V)
    
    # Create simulator
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    
    # Test 1: Sweep input A while input B is low
    print("Simulating with input B = 0V")
    analysis1 = simulator.dc(Vin_a=slice(0, vdd, 0.01))
    
    # Test 2: Sweep input A while input B is high
    print("Simulating with input B = {}V".format(vdd))
    v_in_b.dc_value = vdd@u_V
    analysis2 = simulator.dc(Vin_a=slice(0, vdd, 0.01))
    
    # Test 3: Sweep input B while input A is low
    print("Simulating with input A = 0V")
    v_in_a.dc_value = 0@u_V
    analysis3 = simulator.dc(Vin_b=slice(0, vdd, 0.01))
    
    # Test 4: Sweep input B while input A is high
    print("Simulating with input A = {}V".format(vdd))
    v_in_a.dc_value = vdd@u_V
    analysis4 = simulator.dc(Vin_b=slice(0, vdd, 0.01))
    
    # Test 5: Sweep both inputs A and B simultaneously
    print("Simulating with both inputs A and B sweeping simultaneously")
    # We need to run multiple simulations for this since PySpice doesn't support
    # sweeping two sources simultaneously in a single DC analysis
    sweep_points = 101  # 0 to 1V with 0.01V step
    sweep_voltages = np.linspace(0, vdd, sweep_points)
    output_voltages = []
    
    for voltage in sweep_voltages:
        # Set both inputs to the same voltage
        v_in_a.dc_value = voltage@u_V
        v_in_b.dc_value = voltage@u_V
        
        # Run a DC operating point analysis (no sweep)
        analysis = simulator.operating_point()
        
        # Get the output voltage
        output_voltages.append(float(analysis['out']))
    
    # Plot the results
    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot for sweeping input A
    ax1.set_title('NOR Gate Response - Sweeping Input A')
    ax1.set_xlabel('Input A Voltage [V]')
    ax1.set_ylabel('Output Voltage [V]')
    ax1.grid()
    ax1.plot(analysis1['v-sweep'], analysis1['out'], label='Input B = 0V')
    ax1.plot(analysis2['v-sweep'], analysis2['out'], label='Input B = {}V'.format(vdd))
    ax1.legend()
    
    # Plot for sweeping input B
    ax2.set_title('NOR Gate Response - Sweeping Input B')
    ax2.set_xlabel('Input B Voltage [V]')
    ax2.set_ylabel('Output Voltage [V]')
    ax2.grid()
    ax2.plot(analysis3['v-sweep'], analysis3['out'], label='Input A = 0V')
    ax2.plot(analysis4['v-sweep'], analysis4['out'], label='Input A = {}V'.format(vdd))
    ax2.legend()
    
    # Plot for sweeping both inputs simultaneously
    ax3.set_title('NOR Gate Response - Sweeping Both Inputs')
    ax3.set_xlabel('Input A and B Voltage [V]')
    ax3.set_ylabel('Output Voltage [V]')
    ax3.grid()
    ax3.plot(sweep_voltages, output_voltages, label='Input A = Input B')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Truth table verification
    print("\nNOR Gate Truth Table Verification:")
    print(f"A=0, B=0 => Output ≈ {analysis1['out'][0]:.3f}V")
    print(f"A=1, B=0 => Output ≈ {analysis1['out'][-1]:.3f}V")
    print(f"A=0, B=1 => Output ≈ {analysis3['out'][-1]:.3f}V")
    print(f"A=1, B=1 => Output ≈ {analysis4['out'][-1]:.3f}V")