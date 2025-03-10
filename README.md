# Analog NOR Gate Implementation

This project implements an analog NOR gate using PMOS and NMOS transistors with PySpice.

## Overview

A NOR gate is a digital logic gate that implements logical NOR - it behaves according to the truth table:

| Input A | Input B | Output |
|---------|---------|--------|
| 0       | 0       | 1      |
| 0       | 1       | 0      |
| 1       | 0       | 0      |
| 1       | 1       | 0      |

In this implementation:
- Two PMOS transistors are connected in series between VDD and the output
- Two NMOS transistors are connected in parallel between the output and ground

## Implementation Details

The NOR gate is implemented as a subcircuit using PySpice's `SubCircuitFactory`. The circuit consists of:

1. Two PMOS transistors in series (from VDD to output)
   - When either input is high, the corresponding PMOS is OFF
   - Both inputs must be low for the output to be pulled high

2. Two NMOS transistors in parallel (from output to GND)
   - When either input is high, the corresponding NMOS is ON
   - This pulls the output low

## Requirements

- Python 3.6+
- PySpice
- NgSpice (backend for PySpice)

## Installation

```bash
pip install PySpice
```

Make sure NgSpice is installed on your system.

## Usage

Run the simulation:

```bash
python src/main.py
```

This will run the NOR gate simulation with all possible input combinations and display the truth table.

## Files

- `src/nor.py`: Contains the NOR gate implementation and simulation code
- `src/main.py`: Main script to run the simulation

## Circuit Diagram

```
        VDD
         |
         |
    M1   |
A---|---|
         |
         |---internal
    M2   |
B---|---|
         |
         |---output
         |
    M3   |    M4
A---|---|  B---|---|
         |         |
         |         |
        GND       GND
```

Where M1 and M2 are PMOS transistors, and M3 and M4 are NMOS transistors.