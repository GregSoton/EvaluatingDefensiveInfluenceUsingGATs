# Evaluating Defensive Influence in Multi-Agent Systems Using Graph Attention Networks

Written by Gregory Everett. Email: gae1g17@soton.ac.uk

## Paper Description

Evaluating individual contributions from team members is a critical challenge across many domains, such as security and team sports. While progress has been made in valuing contributions, such as target defence in security or on-ball performance in football (soccer), many aspects of performance, such as off-ball football actions, remain difficult to quantify. We introduce GAPP, a Graph Attention Network model that predicts football pass reception probabilities and provides interpretable insights into off-ball defending. Using attention mechanisms, GAPP captures player interactions and introduces two new metrics to quantify defender contributions. We tested GAPP on 306 English Premier League matches, and showed it reduces binary cross-entropy loss by 6.4 percent compared to multiple baselines for pass reception prediction, while offering unique insights for off-ball defender evaluation for coaches, scouts and teams. This work shows the potential of graph attention networks for analysing complex multi-agent systems like football. The paper for this work was accepted for publication at IEEE DSAA 2025 and will be released soon.

## Ball Reception Prediction

This paper uses a Graph Attention Network model to predict the probability of each attacker receiving the ball at the next event by modelling the game setup as a graph. We show an example plot below of the pass reception predictions.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bb35bae7-c76c-467a-86f7-18816c9060cd" alt="Reception_Prediction" width="600" />
</p>

<p align="center"><em>Predicted pass-reception probabilities for all players at a single event. Marker color/size indicates the model's predicted probability; the highest value shows the model's most likely receiver and the spatial areas with elevated reception likelihood.</em></p>

## Defensive Metrics

The attention mechanism of the GAPP model is used to extract two new defensive metrics for evaluating off-ball defending in football. These metrics are called the Defender Influence and Defender Performance metrics. We provide example plots below of these metrics and explain each of these metrics in the plot description.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b52efd80-d383-41fd-b6ac-a4e5664f9d99" alt="DI" width="500" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/cd54f579-bfe9-4677-befe-a8fbd5a5afa7" alt="DP" width="500" style="margin: 0 10px;" />
</p>

<p align="center"><em>Left — Defensive Influence (DI): the change in an attacker's reception probability when a specific defender's attention is masked. Positive DI indicates the defender reduces the attacker's chance to receive the ball.
Right — Defensive Performance (DP): the DI values aggregated and weighted by each attacker's xT (attacking threat). DP quantifies a defender's overall off-ball positional value in reducing dangerous reception opportunities.</em></p>

## Model Architecture

The architecture of the Graph Attention Network model is provided in the image below. More details on the model hyperparameters are given in the Appendix and the code.

<img width="12925" height="4340" alt="ModelDiagram (2)" src="https://github.com/user-attachments/assets/aadf973e-09ea-41cd-9e22-5ca0f8a32ccf" />

## Use Cases and Findings

## Directory Structure

## Code Workflow

## Data
