Simply run AwarenessCradle.py.


# Meme Neural Network (MNN): A Pathway to Cheaper, More Customized AGI

Based on the scaling law analysis presented in this paper, achieving AGI with simple mechanisms, low cost, and high stability simultaneously proves impossible. To serve the broadest population, we prioritize complex mechanisms over operational costs, aiming to enable personalized AI assistants accessible to all. This section introduces an extensible framework—Meme Neural Network—built upon the Multi-DSL Regulation Model, which implements the Rule-State Duality through meme-based computation. We invite the research community to collaborate on this open-source project[1](@ref).

## Core Concept: Meme as Rule-State Unit

A *meme* represents a continuously activated pathway with specific semantic meaning, manifested as spatiotemporal state-dependent $\mathbf{S}$. Memes are recorded in real-time with non-zero spatiotemporal scales, and all intelligence operations are based on meme manipulations. Each meme can be viewed as an instance of $\mathbf{R}_{\text{learnt}}$, consisting of connectable heads and tails within the $\mathbf{S}$ space, with configurations that dynamically evolve across spacetime.

|![The structure of meme instances $\mathbf{R}_{\text{learnt}}$ in state space $\mathbf{S}$, illustrating connectable heads and tails that enable dynamic recombination. The spatial configuration alone is insufficient to fully characterize a meme, as its identity emerges from rule applications (state interactions) governed by the extraction operator $\mathcal{E}$ and deterministic scale $\mathcal{D}$.](assets/meme_network.png)|
| :---: |
| The structure of meme instances $\mathbf{R}_{\text{learnt}}$ in state space $\mathbf{S}$, illustrating connectable heads and tails that enable dynamic recombination. The spatial configuration alone is insufficient to fully characterize a meme, as its identity emerges from rule applications (state interactions) governed by the extraction operator $\mathcal{E}$ and deterministic scale $\mathcal{D}$. |

## Framework Design Principles

The framework's primary objective is trading mechanism complexity for operational costs: efficiently extracting $\mathbf{R}$ from $\mathbf{S}$, flexibly modifying $\mathbf{R}$, and reducing computational overhead through sophisticated conflict resolution mechanisms.

- **Turing-Complete Expressiveness:** Both individual components and regulatory mechanisms are designed with Turing-completeness to minimize computational costs, ensuring maximum expressiveness within the Rule-State Duality framework.

- **Dynamic Rule Modification:** The framework supports local or collective dynamic modifications of $\mathbf{R}_{\text{narrow}}$ at global, cellular, and connection levels, enabling real-time upgrades.

- **Temporal Dependency Support:** To address computational complexity in temporal modeling, we implement direct time-dependency mechanisms:

 |![](assets/temporal_modeling.png)|
 | :---: |
 
 
  where $y \geq 0$ is meaningful, and $(k, t_s, b)$ are trainable parameters, with $a$ representing weighted activation intensity.

- **Scaling Strategy:** The framework employs a prioritized approach: weight modification first, connection addition second, and cell creation last, preferring rule modifications before space expansions.

- **Flexibility over Parallelism:** Sacrificing parallel training for flexibility, the framework supports hot-updates of rules, dynamic organ cell definition, and external tool integration, relying on multi-core CPU processing over GPU parallelism.

## Learning Dynamics

The framework employs a distributed signal system rather than global reward/punishment functions. Signals trigger computation of task-specific loss for relevant memes and loss for non-productive exploration behaviors.

- **Feedback-based Refinement (Closed-loop):**
  - **Collective Objective:** Complete interaction tasks accurately and efficiently across spacetime while maintaining minimal interaction frequency.
  - **Individual Objective:** Achieve activation tasks accurately and efficiently while minimizing activation frequency.

- **Select-based Search (Open-loop):**
  - **Collective Objective:** Maintain spatiotemporal uniformity and avoid repetitive generation of similar memes.
  - **Individual Objective:** Implement fatigue mechanisms where repeated activation of a connection increases its activation threshold while decreasing thresholds for other connections.

The open-loop and closed-loop modes are mutually exclusive and complementary, and they change dynamically as the hierarchical structure forms and evolves. Therefore, the signal system is also dynamically extensible.

## Intelligence Expression Mechanisms

The framework exhibits intelligent behavior through two complementary mechanisms: individual decision making and systemic regulatory coordination.

- **Individual Intelligence Expression:**
  Individuals demonstrate intelligent behavior by integrating information from multiple relevant memes across spatial and temporal dimensions. This integration enables three key capabilities:
  - **Contextual Activation Decisions:** Individuals evaluate ongoing and historical meme states to determine optimal activation parameters, balancing current context with past experiences.
  - **Adaptive Loss Management:** During loss propagation, individuals assess the relevance of rewards and penalties to their actions, calculating their capacity to absorb loss. They dynamically redistribute loss through source-sink mechanisms to achieve more balanced reward-penalty allocation.
  - **Conflict-Driven Expansion:** When facing activation conflicts, individuals progressively expand their perceptual scope, establish connections with distant memes, and generate new cells to resolve contradictions.

- **Regulatory Intelligence Expression:**
  The regulatory mechanism exhibits intelligence through coordinated management of meme networks:
  - **Distributed Loss Allocation:** The regulator diffuses loss across entire meme pathways to maintain balanced resource distribution.
  - **Complexity-Responsive Scaling:** Loss diffusion increases computational entropy, prompting connections to establish additional links and generate new cells in response to complexity increase.
  - **Temporal Connection Coordination:** The regulator employs temporal priority principles for loss containment, advantage amplification, and exploratory connection strategies.

## Incremental Training Methodology

The framework employs a tutor-student training model where the network initializes as a $\mathbf{R}_{\text{learnt}}$-naive system, gradually acquiring and composing rules through difficulty-progressive tasks. In our experimental framework, the training progresses from basic sensorimotor control (limb state perception) to complex behaviors (standing, lying, running), mimicking biological learning processes. The integration of auditory instructions tests Pavlovian conditioning capabilities, demonstrating how $\mathbf{R}_{\text{learnt}}$ constructs progressively through environmental interaction and task composition.

## Semi-finished Results

Figures 2, 3, and 4 illustrate how the semi-finished MNN, under the signaling mechanism, induces the creation of cells and connections to construct pathways. In the visualization, blue represents the receptor cell interfaces, red represents the effector cell interfaces, and green indicates intermediate cells. Enlarged cells denote activated cells. Blue connections represent physical links, yellow connections signify that a connection is being activated, and green connections indicate that the connection has been successfully activated.

|![Initial State](assets/mnn_demo_fig1.png)|
| :---: |
|**Initial State**|

|![Creating New Cells and Connections](assets/mnn_demo_fig2.png)|
| :---: |
|**Creating New Cells and Connections**|

|![Activating the Effector Cell](assets/mnn_demo_fig3.png)|
| :---: |
|**Activating the Effector Cell**|

[1](@ref): GitHub Links: https://github.com/default01234/Meme-Neural-Network or memenn.com


# Call for Collaboration
This project remains a work in progress. While my original goal was to achieve running capability through training, limited resources have constrained the development. I hope this prototype framework provides value to your research endeavors and welcome contributions from the community.
Contact: coolang2022@gmail.com

No copyright, free to use
