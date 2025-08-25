# Evaluating Defensive Influence in Multi-Agent Systems Using Graph Attention Networks

Written by Gregory Everett. Email: gae1g17@soton.ac.uk

## Table of contents
- Paper Description
- Ball Reception Prediction
- Defensive Metrics
- Potential Use Cases
- Model & Training
- Directory structure
- Code workflow
- Data
- How to cite / contact

## Paper Description

GAPP (Graph Attention for Pass Probabilities) is a graph-attention model that predicts player pass-reception probabilities in football and provides interpretable off‑ball defensive metrics (Defender Influence — DI, and Defender Performance — DP). Trained on 306 EPL matches (359,040 on‑ball events), GAPP reduces binary cross‑entropy (BCE) loss by ~6.4% ± 1.5% compared to baseline models and yields actionable, explainable defensive insights for coaching, scouting and research. The paper for this work was accepted for publication at IEEE DSAA 2025 and will be released soon.

## Ball Reception Prediction

This paper uses a Graph Attention Network model to predict the probability of each attacker receiving the ball at the next event by modelling the game setup as a graph. We show an example plot below of the pass reception predictions.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bb35bae7-c76c-467a-86f7-18816c9060cd" alt="Reception_Prediction" width="600" />
</p>

<p align="center"><em>Predicted pass-reception probabilities for all players at a single event. The highest value shows the model's most likely receiver.</em></p>

## Defensive Metrics

The attention mechanism of the GAPP model is used to extract two new defensive metrics for evaluating off-ball defending in football. These metrics are called the Defender Influence and Defender Performance metrics. We provide example plots below of these metrics and explain each of these metrics in the plot description.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b52efd80-d383-41fd-b6ac-a4e5664f9d99" alt="DI" width="500" style="margin: 0 10px;" />
  <img src="https://github.com/user-attachments/assets/cd54f579-bfe9-4677-befe-a8fbd5a5afa7" alt="DP" width="500" style="margin: 0 10px;" />
</p>

<p align="center"><em>Left — Defensive Influence (DI): the change in an attacker's reception probability when a specific defender's attention is masked. Positive DI indicates the defender reduces the attacker's chance to receive the ball.
Right — Defensive Performance (DP): the DI values aggregated and weighted by each attacker's xT (attacking threat). DP quantifies a defender's overall off-ball positional value in reducing dangerous reception opportunities.</em></p>

## Potential Use cases 
- Post‑match tactical analysis: Visualise event‑by‑event DI/DP to identify moments where defender positioning prevented high‑threat receptions.
- Scouting & recruitment: Rank and compare defenders by normalized season DP to identify players who consistently limit high‑xT receivers.
- Player development & coaching: Provide targeted feedback on off‑ball positioning (which attacker matchups a defender influences most).
- Match preparation / opposition scouting: Identify attackers a defender is particularly effective (or weak) against and adapt marking strategies.
- Research & transfer to other domains: Framework generalises to other multi‑agent systems (e.g., security games or other team sports) where agent influence can be framed via attention.

## Model Architecture

The architecture of the Graph Attention Network model is provided in the image below.

<img width="12925" height="4340" alt="ModelDiagram (2)" src="https://github.com/user-attachments/assets/aadf973e-09ea-41cd-9e22-5ca0f8a32ccf" />

#### Model Details
- Graph construction: fully connected, directed graph with a special ball node (V = players ∪ {ball}).
- Node features: position, velocity, acceleration, distances to goals, distance/angle to ball, on-attacking-team/on‑ball flags.
- Edge features: relative distances, edge angle, difference in node angles to the ball, same‑team flag.
- GAT specifics: 2 GAT layers, 16 attention heads, skip connection + concatenation to encoders.

#### Model Hyperparameters and Training
- Batch size: 64; epochs: 200.
- Node hidden size: 32; edge hidden size: 16.
- Loss: Binary cross‑entropy (attacking players masked to compute loss only where relevant).
- Optimiser: Adam, initial LR = 0.003.
- Dataset used: 306 EPL matches (2023/24) → ~359,040 on‑ball events.

## Directory Structure

## Code Workflow

## Data

The EPL Premier League Events and Tracking Data was supplied by Gradient Sports, who supported this work. For enquiries about this or similar datasets to test this work, please visit: https://www.gradientsports.com/.

## How to cite / contact
If you use this work, please cite:

“Evaluating Defensive Influence in Multi-Agent Systems Using Graph Attention Networks”. Gregory Everett, Ryan J. Beal, Tim Matthews, Timothy J. Norman, and Sarvapali D. Ramchurn (2025). In: 2025 IEEE 12th International Conference on Data Science and Advanced Analytics.

Contact: gae1g17@soton.ac.uk
