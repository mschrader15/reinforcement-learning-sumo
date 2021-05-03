# Optimizing Traffic Signal Timing in SUMO w/ Reinforcement Learning

## Project Proposal

### Background

The DOE project is a joint venture between the University of Alabama (UA) and Oak Ridge National Laboratory, with the aim being improved network-wide fuel economy. Initial findings from a test corridor in Tuscaloosa show that improved light timings correspond to a 2-12% reduction in fuel consumption. As part of the project, a partnership with the German Aerospace Center (DLR), and specifically their SUMO traffic simulation software, has been created giving UA the tools to perform microscopic traffic simulations of the Tuscaloosa region and to test potential light timing improvements. SUMO has a built-in vehicle model, but the ability of that model to accurately capture vehicle dynamics is under question. To get a better estimate of energy efficiency improvements, Oak Ridge and UA are partnering to develop more detailed vehicle models.

The optimization of traffic signal timing has been researched extensively in the past, but this research typically focuses on increasing vehicle average velocity, maximizing vehicle flow or decreasing waiting time. Most of the research in combining reinforcement learning (RL) and traffic simulation has been in influencing connected vehicle and autonomous vehicle (AV) control, with one of the classic examples being RL controlled AV&#39;s dissipating traffic waves and bottleneck decongestion.

My project will focus on combining the SUMO micro-simulation vehicle emissions models and traffic signal optimization, using RL to potentially achieve significant improvements in overall system energy efficiency.

### Environment Model

The environment model is a SUMO model of three intersections in Northport, Alabama. The three intersections serve industrial, commercial (Walmart, Lowes) and residential areas, respectively. The traffic in the simulation has been carefully calibrated to match real-world traffic as close as possible. Calibration is done by aggregated detector counts (stored in a SQL server) and deriving traffic routes that satisfy the counts most closely. To benchmark the RL-controlled traffic signals, the simulation will be run with the lights behaving exactly the way that they did on any given day.

The observation space for the RL agent includes the count of vehicles in each lane approaching the intersection, with some maximum distance that the &quot;sensor&quot; can pick up cars at, as well as information about the other traffic signals&#39; states. The action space is three discrete numbers, which represent the possible combinations of the traffic signal&#39;s phases. The reward is the [Fuel Consumption Intersection Control Performance Index](https://journals.sagepub.com/doi/abs/10.1177/03611981211004181).

### Approach

Custom code will be written to create an OpenAI environment wrapping SUMO, and the RL agent will be trained using [RLLIB](https://docs.ray.io/en/master/rllib.html). According to results in [the Flow benchmarks](https://flow-project.github.io/papers/vinitsky18a.pdf), Evolutionary Strategies will be used to train the traffic signal controller.

## Final Report

### Procedure

The process of training a reinforcement learning (RL) agent to control three traffic signals can be divided into four major parts: creating a SUMO network, generating traffic demand and following traffic signal states, creating an environment for the RL algorithm, and training the RL algorithm. The first two were completed prior to the start of this project, but a brief overview will be presented below.

#### SUMO Network

Traffic simulation starts with creation of a network. In SUMO, the network file contains the definition of the road geometry, lane layout, intersection type, and traffic signal parameters. Figure 1 shows the completed network model, with the intersections labelled TL1-3, from west to east. It is important to note that this traffic network lies to the north of Tuscaloosa, Alabama and represents a portion of US82, which connects Tuscaloosa to smaller towns to the west.

![](RackMultipart20210503-4-194p25e_html_568df6bdacaa8d55.png)

_Figure 1. SUMO Network_

#### Demand Generation + Traffic Signal Following

Next in the traffic simulation workflow comes demand generation. Effectively comparing the RL controlled traffic signals to those of the real-world requires traffic demand to be as close to realistic as possible, as intersection performance metrics are based on vehicle-based metrics. To create realistic traffic demand, real-world detector counts on the representative day (February 24th, 2020) were aggregated and corresponding route definitions were generated using SUMO&#39;s [routeSampler](https://sumo.dlr.de/docs/Tools/Turns.html#routesamplerpy) tool. (Route definition is a SUMO term corresponding to the streets a vehicle traverses and the time it enters the network). It is not a one-shot process, and many iterations were required to meet USDOT calibration metrics.

Lumped in with demand generation is writing code to copy the light states of the target simulation day. The same database that stores the detector events also saves a history of traffic signal actions to a precision of 100ms, and thus it is possible to accurately copy the actions of the traffic signals. The ability to replay both the traffic signals and the traffic flows creates a crucial baseline to which the RL controlled traffic signals can be compared.

All demand generation and traffic signal coding occurred prior to the start of this project, so further explanation will be excluded. The source code is available in the [airport-harper-sumo](https://github.com/UnivOfAlabama-BittleResearchGroup/airport-harper-sumo) repository on GitHub.

#### The Environment

This project picked up on the third part of the procedure: generating an environment for the RL algorithm to plug in to.

The original plan for this project was to utilize [Flow](https://github.com/flow-project/flow), but further analysis into its codebase made it clear that it would be hard to use with the airport-harper-sumo repository. Flow abstracts all the SUMO XML inputs, which the airport-harper-sumo repository uses almost exclusively. Not wanting to rewrite route generation and traffic-light-following code (and knowing that SUMO runs faster using XML inputs), the decision was made to write a new environment.

The environment borrowed from the Flow project in some areas, but it is essentially a custom wrapper around the [OpenAI Gym](https://gym.openai.com/). It has an observation space, an action space and a reward function. The Gym also dictates what happens when the RL algorithm takes a &quot;step&quot; as well as &quot;resetting&quot; between training iterations. Of interest for the report are the observation space, action space, and reward function, but all of the code can be found in the [reinforcement-learning-sumo](https://github.com/mschrader15/reinforcement-learning-sumo) repository on GitHub. The code is thread-safe and can be parallelized, as well as supporting policy replay with visualization and customizable outputs.

##### Observation Space

The observation space is a combination (OpenAI Tuple) of four arrays. The first array represents the number of cars in each lane approaching the three intersections in the network and _roughly_ emulates a camera sitting in the center of the intersection. In practice, a circle with a radius of 100 meters is drawn from the center of each intersection and any vehicle inside that circle is added to its respective lane count. The next array reports each traffic signal&#39;s state, which is enumerate as an integer pointing to the corresponding traffic signal state in a [SUMO traffic signal description file](https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html#defining_new_tls-programs). The third array reports an enumerated value of the traffic signal&#39;s current color and the fourth is the time that each traffic signal has been in a green phase. It can be summarized as:

##### Action Space

The goal of the action space was to emulate the way that an implemented artificial neural network (ANN) would interact with the lights. In the real world, the traffic signals cannot simply switch when requested, instead they must wait until internal count-down timers clear. Therefore, to place a restriction on the RL actor, the environment&#39;s actor class implements the countdown timers. The RL agent can take up to eight actions at the three traffic signals, each corresponding to a movement of the traffic signal, for example serving phases 2 &amp; 6. When the RL algorithm requests a phase transition that is not allowed or before the count-down timers clear, the request is ignored. The action space can be summarized as below:

##### Reward

The reward function for this project was the Fuel Consumption Intersection Control Performance Index (FCIC - PI). Specifically, the FCIC-PI is a metric proposed by Stevanovic, Shayeb, &amp; Patra in the aptly named paper [Fuel Consumption Intersection Control Performance Index](https://journals.sagepub.com/doi/abs/10.1177/03611981211004181) to incorporate fuel consumption into the common Performance Index, which results in the following equation:

where is a road-specific scaling factor, assessing a penalty to stopped vehicles, based on the resulting fuel consumption penalty of re-accelerating the vehicles. It is formally defined as

where is the fuel consumption of one vehicle. Stevanovic, Shayeb, &amp; Patra present empirically calculated values for a road network in Florida, and these same values were used in the reward function implemented for this project. Vehicle delay is calculated by comparing each vehicle&#39;s actual speed to the allowed speed of the edge. Further information can be found in the [SUMO documentation](https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html#retrieve_the_timeloss_for_all_vehicles_currently_in_the_network). The number of stopped vehicles is the count of all vehicles in the network with a speed less than 0.3 m/s.

To introduce stability, the reward at a given simulation step is calculated as the moving average of the past 10 simulation seconds of rewards. The choice of 10 seconds is completely arbitrary. Going forward the role that the moving average plays in the result should be investigated.

#### RL Algorithm + Training

Training the RL agent was the real focus of this project, but CPU time aside, it was the least time-consuming part. Reinforcement learning algorithms were tested on the environment using the [ray RLlib library](https://docs.ray.io/en/master/rllib.html). Ray is a distributed computing library that handles the overhead required for parallelizing training rollouts and optimizing policies. RLlib has many different reinforcement learning algorithms but, according to results in [the Flow benchmark](https://flow-project.github.io/papers/vinitsky18a.pdf) paper, only evolutionary strategies (ES) and proximal policy optimization (PPO) were trialed for this project. The authors found that gradient-free methods, specifically ES, perform the best for traffic signal optimization. PPO was applied to the environment to compare ES to a more classical RL method.

##### Evolutionary Strategies

ES is a black box optimization algorithm method that is not based on the Markov Decision Process (MDP). [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/pdf/1703.03864.pdf) lays out several benefits for ES over MDP-based RL algorithms, but the fact that it performs well in environments with long time-horizons and complicated reward structures is important to traffic simulation, specifically traffic signal optimization. As presented in the previous section, the reward used for this project is a rolling average derived from vehicle behavior. The traffic signals certainly influence vehicle behavior, but so do interactions between other vehicles. The nature of traffic complexities makes the reward inherently complicated and, as the results section will show, ES proved the only method that converged on a solution.

## Results

The results presented in this section represent the findings from applying the ES and PPO RL algorithms to the [airport-harper-sumo](https://github.com/UnivOfAlabama-BittleResearchGroup/airport-harper-sumo) SUMO model. Traffic was simulated starting at 6:30 AM on the target day (Feb. 24th, 2020), then the network underwent an hour-long warmup period until the RL algorithm took over at 7:30 AM. The RL algorithm was then trained over 15 minutes of simulation, until 7:45 AM. All training was done on a Linux machine running Ubuntu 18.04 with two AMD 32-core processors.

### Algorithm Comparison

Both ES and PPO were used to train an artificial neural network with three hidden layers (100, 50, 25). The training hyper parameters were taken from [the Flow benchmarks.](https://flow-project.github.io/papers/vinitsky18a.pdf)They were trained for 500 iterations with 64 rollouts per iteration.

Figure 2 below shows the average episode reward for the two different algorithms considered. It clearly shows that PPO does not converge to a solution, whereas the ES algorithm does more than 80% of its learning in the first 10% of the training period and converges. A video of the ES training iterations plotted below has been uploaded to [YouTube](https://youtu.be/wDe6mTLmpL4).

![](RackMultipart20210503-4-194p25e_html_eaa0bcfe006691.png)

_Figure 2. Evolutionary Strategies vs. Proximal Policy Optimization_

### RL vs Real World

After training the RL agent, various comparisons were done between the real-world traffic signals and the RL-controlled ones, specifically the ES algorithm from above, as the PPO algorithm did not converge to a reasonable policy. Before the analysis begins, it is important to note that the real-world traffic signals function in simulation like static traffic lights, meaning they switch according to a time (corresponding to the date and time they switched in the real world) and not in reaction to simulation traffic. While the simulated traffic is calibrated to match the actual traffic as best as possible, the fact that the that the signals are not reactive puts them at a disadvantage to the RL trained traffic signals. An interesting research opportunity is to pair [software-in-loop traffic controllers](https://youtu.be/28x2Iye3FRM) with this calibrated simulation and conduct a more realistic comparison of RL vs. real-world.

Figure 3 shows the rolling reward over 15 minutes of simulation time. The RL agent stabilizes the reward more effectively than the real-world traffic signals, and clearly is not subject to the same dips that the real-world traffic signals cause. These dips correspond to periods when the side streets are being served by the real-world traffic signals.

![](RackMultipart20210503-4-194p25e_html_6229261f4cb72b9e.png)

_Figure 3. Reward at each step during simulation_

A more detailed analysis of the simulation reveals why the reward function of the RL controlled network is better than the real-world reward. Figure 4 below illustrates some of the metrics influencing the reward function.

![](RackMultipart20210503-4-194p25e_html_79c276b079863530.png)

_Figure 4. Various Network Performance Metrics over Simulation Period_

The RL algorithm does a better job at smoothing all network metrics and scores better on average. It most drastically cuts down the network waiting time, which is likely due to the reward function penalizing the number of stopped vehicles with the factor. The average speed in the network is almost uniformly higher. The average MPG for vehicles in the network is higher as well, however only slightly.

Figure 5 and Figure 6 on the next page highlight the difference between the real world and RL controlled traffic signals. In these figures, dark green corresponds with a green phase, while light green is a yielding or flashing left turn. In Figure 5, the traffic light pattern is easy to spot: the traffic lights are coordinated, and they have a fixed cycle length. In Figure 6, on the other hand, there is no pattern. The RL agent is reacting to traffic flows and making decisions to maximize reward. It is interesting that it can switch so rapidly between phases and still maintain high average network speeds, but Figure 4 shows that it is successful.

![](RackMultipart20210503-4-194p25e_html_68dfc34a9df54286.png)

_Figure 5. Real World Traffic Signal Green &amp; Yield Phases (TL1&amp;2 Continue to operate past 7:40)_

![](RackMultipart20210503-4-194p25e_html_e5e283159942fd90.png)

_Figure 6. RL Controlled Traffic Signal Green &amp; Yield Phases_

## Conclusions

Creating [reinforcement-learning-sumo](https://github.com/mschrader15/reinforcement-learning-sumo) was a lot of fun! Working on this project created an extensible codebase and baseline to carry over into the DOE Project. It will be interesting to see how the different reward functions change the trained outcomes going forward, and what those outcomes look like.

One clear conclusion of this project was that ES is at the very least a more forgiving reinforcement-learning framework. By simply copying the hyperparameters used in the Flow benchmarks, a solution was converged upon. With PPO, there was no such luck. That being said, more RL algorithms should be trialed with various hyper parameters before concluding that ES is the definitive way forward.

Not presented in the results was an attempt at transfer learning, where ANN trained in Figure 2 over the 7:30 – 7:45 time interval was transferred to 17:30 - 17:45 and trained for another 500 iterations. The final solution was converged upon in very few iterations and mirrored the results from training the RL agent on the 17:30 - 17:45 interval from scratch. It would be interesting to investigate transfer learning on the network further, as it could potentially save considerable computation time vs. training on longer time intervals. SUMO slows down significantly when traffic congestion increases, which happens in early RL training iterations. Longer simulations mean more congestion, so if transfer learning could be used instead, that would likely be more computationally efficient.
