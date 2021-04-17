# Reinforcement Learning + SUMO
Applying reinforcement learning to SUMO for ME-691

## Structure
This project follows the structure of ![FLOW](https://github.com/flow-project/flow) closely. I only chose to diverge from FLOW because it abstracted the XML creation for SUMO. For me, this repository plugs in to a greater code-base, that turns real-world ITS data into SUMO traffic demand and traffic light operation. Those tools work ultimately by writing to SUMO XML files, and we didn't want to convert to the FLOW pythonic framework.

The project copied/referenced the ![sumo-rl](https://github.com/LucasAlegre/sumo-rl) repository.

## Results

