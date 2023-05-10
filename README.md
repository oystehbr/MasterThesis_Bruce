# Master's thesis - Øystein Høistad Bruce 🥇
Supervisors: Signe Riemer-Sørensen, Øyvind Ryan and Vegard Antun

Folders:
- __reinforcement_learning__ 🤖: the code for the reinforcement learning part.
- __fuel_model__ ⛽︎: the code for the fuel model part. 

<h2 align="center">Simulation of the best plan obtained by the framework, on an arbitrary map</h2>

![Alt Text](reinforcement_learning/transport/animations/map5x5_V1_plan.gif)

<h2 align="center">Abstract</h2>

Road construction sites are often inefficient, with construction machines frequently idling for extended periods, wasting fuel and time.
One way to increase efficiency is to optimize the scheduling of the dumpers transporting materials across the site. 
This thesis proposes and investigates a multi-agent reinforcement learning framework designed to coordinate dumpers and excavators on construction sites. 
The framework generates time schedules for all vehicles, considering multiple criteria such as fuel consumption, completion time, and cost. 
Users can choose a plan that best aligns with their preferences, ensuring maximum efficiency. 
The framework has a negligible training time and generally outperforms a baseline constructed from human behavior.
In addition, we developed a predictive fuel consumption model for a dumper using high-resolution data logged over 12 working days.
By associating each dumper with such a model, we can more accurately predict their fuel consumption while driving, further improving the planning. 
