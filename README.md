# TransitGraphAI
> *Work in progress – code will be released soon.*
This project explores the intersection between **urban mobility**, **graph representation learning**, and **autonomous decision-making**.  
It combines **generative AI** and **reinforcement learning** to model and simulate realistic public transport networks and traveler behaviors.

---

## Project Overview

The project is based on an extensive sampling of **GTFS data** (General Transit Feed Specification) from **16 European public transport networks**.  
After collection, the data was reformatted to build **time-dependent multimodal transport graphs**, including routes, interconnections, and schedules.

Two main research directions are being developed in parallel:

### (1) Generative Modeling of Transport Graphs
I experimented with **graph generative models** such as:
- **Variational Autoencoders (VAE)**  
- **VAE-GAN hybrids**

The objective is to generate **synthetic transport networks** that capture structural properties of real urban systems (connectivity, modularity, hub density, etc.).  

For the initial proof of concept, models were trained on **Erdős–Rényi random graphs**, as real GTFS-derived adjacency matrices are currently too large for direct training.  
Future iterations will include a compressed graph representation to handle real-world data efficiently.

---

### (2) Autonomous Agent for Route Optimization
In a second part of the project, I prototyped an **autonomous agent** trained via **Proximal Policy Optimization (PPO)**.  
The agent learns to **find the fastest route between two stations**, given:
- A **departure station and time**
- Limited local knowledge of the network (no full map access)
- Dynamic constraints from transport schedules

The goal is to explore how **reinforcement learning** can approximate human-like route-finding behavior under uncertainty.

---

## Goals and Perspectives

- Create **synthetic transport networks** preserving real-world topological and temporal patterns  
- Simulate **adaptive navigation strategies** in dynamic environments  
- Build a bridge between **graph generation**, **transport modeling**, and **reinforcement learning**


