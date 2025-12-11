Simply run AwarenessCradle.py.


# Meme Neural Network (MNN): A Pathway to Cheaper, More Customized AGI

Based on the scaling law analysis presented in [A Unified Intelligence Theory based on Rule-State Duality and its Multi-DSL Regulation Model](https://download.ssrn.com/2025/11/5/5668911.pdf?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjECYaCXVzLWVhc3QtMSJIMEYCIQCisNYxiStiUW%2B8e7%2BpbAZdE0ZG6tia%2BVwBZnAcOhlhmQIhAMk0jWS7ZL8JmSnlosBAh2mfH8R2vA9FJUUSMRynqa6qKsUFCO%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMMzA4NDc1MzAxMjU3IgwsHMUIKkGoAwpfxNkqmQXQsWOjQloZaE1ZJ3CMuRgLz3xGyGh%2FFknAYXSIvoqUjoNuqYJ%2BM43VF%2FuYgOc1nELupIiGUXPQkx%2FBehz5%2ByfZgzQ09IYyYPP78VljF5WOEYPLGcm4cC4eEXnCMxwIp1YamR6JDFiAO8Ix8RYnWp8o0vE6mnmCIb7UA9xQLFoGF7Xcirpvqzt5h97kW44IieNnbLLrzK4cL5RFRWP7jry3DCyCQTZjfr6%2FjZnFlIQg1vbJjjQJDf4zYcORFFeTVMmup1au7IN1k6bkWjnv8iF%2FJgFHK69dq4%2Bl1uL1czu4EOsL6tVuUdIPmWc21AU11yAWwL7cBcanPpPLToKE%2BlztmxWYD5juKQimU0fAAyqLhvbYXDikPeoN4rC3Ghr9rvakFlj%2B7Vtv1Juj7kjWGIDHBT7Be2W%2BIK8wNYsvhbNGP%2F965gx%2BWId9jKkmoxv8SB%2FgIwXMkz2unWfRROFBynZ8HshSTutsbFLFxpzITSkf3ksmpQAggPBNhQo1907mWj%2FRcjl8tDX6xla%2Fhwyi1ArJuZkiAJHKbaJvGz9c%2BgI94xMkvaHVLpKNEmL%2F9gvCJCFeqjK40x3CjlK2SHILf0NjGNO46CgJVcCKOhDJKMMZlM8otCpKS2s8fKjKMyS8ZhihDgNb%2F3qClWulvYzTiIuJjaWpnJdEVI7yNsDS2dS42w%2FWzYlDj%2BQtCx8nzYO7rc9P2E2ymV8O48UjG0dZZePfavupADNP1JvowBvBWWqV882yWrWrG%2FZAibvxiodn37C%2BE6jrpIR7Q6PvKtug4SjJAXa2GofymWMzY50wR8%2Bgl%2BaffKaHfvlhD%2F3nXySHkyBF0rRqV5q0DG6UP%2FYLokTHZG87kUOnIK%2BZpT%2FttnAxCBOBQqMu1mLArzDanOvJBjqwAUwtRD9HUrh%2FEnTrFqegyOvb0Rv3X4HRKAky5kepOVu4Wtw29tOF1hzVtOudEsGxW4yQBeihlocmN1dcIW3%2F8dLhuqni6vJbv%2Bqn%2BlnDX6VWVLzC4LaVURKYnvQb3kbWKilu1bpKA5FyXjQ5fK4X%2FEy4IvtrixTM0MdwlrJdajHd2lywsqfgUVwmQccZs0kyBbdDjW2X80UCkLvsM17VB%2FdBV%2Fsp743YQ0HeN2QqR84x&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20251211T150305Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAUPUUPRWEZNZR4S3M%2F20251211%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=898e6f9ff60aa56a70f3623a4f5927990834a4184634d6a5c759e0f101ec8499&abstractId=5668911), achieving AGI with simple mechanisms, low cost, and high stability simultaneously proves impossible. To serve the broadest population, we prioritize complex mechanisms over operational costs, aiming to enable personalized AI assistants accessible to all. This section introduces an extensible framework—Meme Neural Network—built upon the Multi-DSL Regulation Model, which implements the Rule-State Duality through meme-based computation. We invite the research community to collaborate on this open-source project[1](@ref).

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

The framework employs a tutor-student training model where the network initializes as a $\mathbf{R}_{\text{learnt}}$-naive system, gradually acquiring and composing rules through difficulty-progressive tasks. In our experimental framework, the training progresses from basic sensorimotor control (limb state perception) to complex behaviors (standing, lying, running), mimicking biological learning processes. 

The integration of auditory instructions tests Pavlovian conditioning capabilities, demonstrating how $\mathbf{R}_{\text{learnt}}$ constructs progressively through environmental interaction and task composition.

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
