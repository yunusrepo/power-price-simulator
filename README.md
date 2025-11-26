# Power Price Simulator

This repository contains a continuous time power price simulator with:

- A two state Markov regime (calm and stressed)
- Heston style stochastic variance
- A jump component for price spikes
- Monte Carlo paths and basic analytics

The goal is to illustrate how I approach modelling of stylised power prices
in a clean, modular way using only synthetic data and generic parameters.

## Structure

- `src/config.py`  
  Parameter dataclasses for the model and simulation.
- `src/regime_switch.py`  
  Simulates a discrete time Markov chain for regimes.
- `src/heston_jump_diffusion.py`  
  Simulates a single price and variance path with Heston dynamics and jumps.
- `src/simulator.py`  
  Runs multiple paths and collects them into a DataFrame.
- `src/analytics.py`  
  Basic analytics: summary statistics and percentiles.
- `scripts/run_simulation.py`  
  End to end script to generate paths, plot samples and save outputs.

All data is synthetic and calibrated only for demonstration.

## Quick start

Install requirements and run:

```bash
pip install -r requirements.txt
python -m scripts.run_simulation
