# Reinforcement Learning + SUMO
Applying reinforcement learning to traffic microsimulation (SUMO)

**Very much a WIP**

## Structure
This project follows the structure of [FLOW](https://github.com/flow-project/flow) closely. I only chose to diverge from FLOW because it abstracted the XML creation for SUMO. For me, this repository plugs in to a greater code-base, that turns real-world ITS data into SUMO traffic demand and traffic light operation. Those tools work ultimately by writing to SUMO XML files, and we didn't want to convert to the FLOW pythonic framework.






## Results

Full breakdown: https://maxschrader.io/reinforcement_learning_and_sumo

video [on Youtube](https://youtu.be/wDe6mTLmpL4)

### ES vs. PPO
![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Fmax-schrader%2F4_odcf9vT1.png?alt=media&token=fe2c0a84-e5de-406c-8425-2af9cb7b1498)


### RL-Controlled Traffic Signals vs. Calibrated Real-World Simulation 
![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Fmax-schrader%2Fg0bsQ-CUPL.png?alt=media&token=0f6a7680-8a0b-4a59-983d-8ddf0dd1908c)

### 

#### Real World Traffic Signals during Simulation Period

![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Fmax-schrader%2FbGeusZyZMr.png?alt=media&token=2fce5398-1604-4d4c-af74-8ffbd1c07c98)

#### RL-Controlled Traffic Signals during Simulation Period

![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Fmax-schrader%2FyAe6vO9VhL.png?alt=media&token=d05283d2-253c-45f3-a912-fd5c34147db8)






