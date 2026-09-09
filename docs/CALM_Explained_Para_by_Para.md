# CALM: Conditional Adversarial Latent Models for Directable Virtual Characters
## Complete Paragraph-by-Paragraph Technical Analysis & Guide for Humanoid Robotics (Unitree G1)

---

### Executive Overview & Paper Metadata
* **Authors:** Chen Tessler, Yoni Kasten, Yunrong Guo, Shie Mannor, Gal Chechik, Xue Bin Peng
* **Institutions:** NVIDIA, Technion Institute of Technology, Bar-Ilan University, Simon Fraser University
* **Venue:** ACM Transactions on Graphics (SIGGRAPH 2023)
* **ArXiv Reference:** [arXiv:2305.02195](https://arxiv.org/abs/2305.02195)
* **Target Robotics Platform for this Guide:** Unitree G1 Humanoid Robot (Hierarchical Behavioral Control / HBC Framework)

```mermaid
flowchart TD
    subgraph Phase 1: Low-Level Pre-Training
        M[MoCap Sub-Sequence M] -->|Raw Kinematics| E[Motion Encoder E]
        E -->|Latent Vector z| Pi[Low-Level Policy \pi(a|s, z)]
        E -->|Latent Vector z| D[Conditional Discriminator D(s, s'|z)]
        Pi -->|Micro-Actions a (PD Targets)| Sim[Physics Simulator (Isaac Gym / G1 Sim)]
        Sim -->|State Transitions (s, s')| D
        Sim -->|State Transitions (s, s')| Pi
        RefData[Reference Dataset \mathcal{M}] -->|Demonstrations (s_hat, s'_hat)| D
        D -->|Adversarial Reward r_t| Pi
    end

    subgraph Phase 2: Precision Training
        Goal[Task Goal / Direction d*] --> HighPi[High-Level Policy \pi_high]
        StyleRef[Style Motion z_hat] --> HighPi
        HighPi -->|Directable Latent z_t| Pi_Frozen[Frozen Low-Level Policy \pi]
        Pi_Frozen --> Sim2[Simulated Humanoid / G1]
    end

    subgraph Phase 3: Zero-Shot Task Execution (Inference)
        FSM[Finite State Machine / Game Logic] -->|Sequence of Styles & Directions| HighPi_Frozen[Frozen High-Level Policy]
        FSM -->|Direct Skill Triggers| Pi_Frozen
        HighPi_Frozen --> Pi_Frozen
        Pi_Frozen --> RealG1[Unitree G1 Robot]
    end
```

---

# PART I: FRONT MATTER & INTRODUCTION

---

## 0. Abstract

### Paragraph 0.1 (Abstract Text)
> **Original Paper Text:**
> *"In this work, we present Conditional Adversarial Latent Models (CALM), an approach for generating diverse and directable behaviors for user-controlled interactive virtual characters. Using imitation learning, CALM learns a representation of movement that captures the complexity and diversity of human motion, and enables direct control over character movements. The approach jointly learns a control policy and a motion encoder that reconstructs key characteristics of a given motion without merely replicating it. The results show that CALM learns a semantic motion representation, enabling control over the generated motions and style-conditioning for higher-level task training. Once trained, the character can be controlled using intuitive interfaces, akin to those found in video games."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **The Fundamental Dilemma in Physics-Based Character / Humanoid RL:**
  Prior methods suffered from a sharp trade-off:
  1. *Tracking-based Imitation (e.g., DeepMimic):* Highly realistic tracking of a *single* motion clip, but completely brittle, cannot generalize, and scales poorly (requires $N$ separate policies for $N$ clips).
  2. *Unconditional Adversarial Imitation (e.g., AMP - Adversarial Motion Priors):* Trains a policy on a multi-motion dataset, but the discriminator only judges if a transition looks "human-like" in aggregate. The agent picks one or two degenerate behaviors (mode-collapse) to solve a task, ignoring the rest of the dataset.
  3. *Unsupervised Skill Embeddings (e.g., ASE - Adversarial Skill Embeddings):* Maximizes mutual information between arbitrary latents $z \sim p(z)$ and state transitions. While diverse, the resulting latent space is disorganized: you cannot give it a specific motion (e.g., "crouch walk") and know which $z$ produces that motion. Controllability is sacrificed for diversity.
* **CALM's Breakthrough:**
  CALM solves this by jointly learning:
  - A **Motion Encoder** $z = E(M)$ that projects any motion sequence $M$ into a continuous, semantically structured latent space $\mathcal{Z} \subset \mathbb{S}^{d-1}$.
  - A **Conditioned Low-Level Policy** $\pi(a|s, z)$ that acts as a universal motion decoder.
  - A **Conditional Discriminator** $\mathcal{D}(s, s' | z)$ that forces the policy to specifically imitate the motion encoded in $z$, rather than any arbitrary human motion.
* **Key Concept: Reconstruction vs Distributional Generalization:**
  Notice the phrase *"without merely replicating it"*. Unlike Variational Autoencoders (VAEs) that penalize exact joint-angle L2 tracking error, CALM uses an adversarial objective. The policy learns the *underlying motion distribution* of the skill. This allows the humanoid to adapt foot placement, balance against physical perturbations, and step over obstacles while preserving the core style.

#### 2. Unitree G1 Humanoid Control Relevance
* **Why G1 Needs This:**
  Training a 23-DOF or 29-DOF humanoid like the Unitree G1 using raw task rewards (e.g., velocity tracking) results in unnatural "robot-like" gaits: bent knees, jerky arm twitches, high joint torques, and excessive foot impact forces that damage physical actuators.
* **The CALM Role on G1:**
  CALM serves as the **Low-Level Motor Prior (LLC)** in a Hierarchical Behavioral Control (HBC) stack. Instead of training G1 to walk, jump, crouch, or push boxes from scratch, G1 learns a universal motor manifold from large MoCap datasets (AMASS, CMU). A high-level task policy (or teleoperator) simply sends a 64-dimensional latent vector $z$, and the low-level policy translates it into stable PD joint targets ($q_{\text{target}}$) at 30–50 Hz, preserving natural bipedal dynamics and balance.

---

## 1. Introduction

### Paragraph 1.1 (The Challenge of Realistic Humanoid Movement)
> **Original Paper Text:**
> *"Virtual environments and interactive characters have become more prevalent and user-friendly, but creating realistic and diverse behaviors for these virtual agents remains a challenge due to the complexity of human motion. To create interactive and immersive experiences, virtual agents must adapt to different environments and user inputs in a life-like manner, and this requires the ability to perform a wide range of behaviors on demand. To that end, we need to develop control models that can generate complex and realistic behaviors, while taking into account the properties of the environment. For example, in virtual reality games, players that interact with virtual characters and objects expect them to behave realistically. This includes responding to user commands and navigating through virtual environments. When virtual agents fail to respond naturally to user input, it can disrupt the immersive experience."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* Human motion is high-dimensional, highly non-linear, and inherently dynamic. A humanoid has high degrees of freedom (DOFs), unilateral contact constraints (feet can only push against the ground, not pull), non-linear Coriolis and centrifugal forces, and continuous contact mode switches (single support vs double support).
* Traditional kinematic animation (motion graphs, blend trees) produces visual foot sliding, cannot react to external forces, and violates physical laws (gravity, friction, inertia).
* Physics-based simulation solves the physical plausibility, but controlling the simulated body dynamically on demand across dozens of distinct skills remains exceptionally hard.

#### 2. Unitree G1 Humanoid Control Relevance
* Replace "virtual character in VR" with "Unitree G1 in human environments (homes, warehouses)".
* The G1 has 23 actuated joints (6 per leg, 4 or 7 per arm, 1 to 3 in the torso). When interacting with humans or uneven terrain, hand-crafted state machines or single-gait controllers fail to adapt. The robot must dynamically shift between standing idle, walking at varying speeds, crouching to avoid obstacles, and stepping sideways to maintain balance.

---

### Paragraph 1.2 (Limitations of Prior Art: DeepMimic & ASE)
> **Original Paper Text:**
> *"Recent advancements in machine learning and access to high-quality human motion capture data have led to the development of control policies that can replicate human behavior. Early studies in this field, such as [Peng et al. 2018], focused on imitating single motion clips. However, as each motion is learned using an independent controller, it does not effectively scale. Later research [Peng et al. 2022, ASE] aimed to improve the diversity of generated motion by learning a latent-conditioned controller and maximizing a mutual information objective. When trained on a dataset of diverse motions, distinct behaviors emerged, however at the cost of losing the ability to control the generated motion."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **DeepMimic (Peng et al. 2018):** Uses explicit tracking rewards:
  $$r_t = w_p r_t^p + w_v r_t^v + w_{ee} r_t^{ee} + w_{com} r_t^{com}$$
  tracking target joint angles $q^*_t$, joint velocities $\dot{q}^*_t$, end-effector positions $x_{ee}^*$, and center of mass $x_{com}^*$.
  * *Flaw:* Only tracks one clip at a time. If you have 500 clips in your dataset, training 500 individual controllers is completely unscalable.
* **ASE (Adversarial Skill Embeddings - Peng et al. 2022):** Formulates an unsupervised objective maximizing the Mutual Information $I(S'; Z)$ between next state $S'$ and latent skill variable $Z$, alongside an unconditional AMP discriminator:
  $$\max_\pi \mathbb{E}_{z \sim p(z)} [r_{\text{AMP}}(s, s') + \beta \log q(z | s')]$$
  * *Flaw:* The mapping from $Z$ to behaviors is purely unsupervised. $z = [0.2, -0.9, \dots]$ might trigger a backflip, while $[-0.1, 0.4, \dots]$ triggers a crawl, but the latent space has no semantic organization. You cannot query: *"give me the latent vector for a left turn"*. Thus, directability is lost.

#### 2. Unitree G1 Humanoid Control Relevance
* On the Unitree G1, you cannot afford random behavior exploration during deployment. A high-level navigation controller needs predictable, reliable low-level execution: commanding "crouch walk at 0.5 m/s" must reliably trigger that specific locomotion gait without the robot suddenly deciding to kick or roll over.

---

### Paragraph 1.3 (Introducing CALM)
> **Original Paper Text:**
> *"Building on these previous works, we present Conditional Adversarial Latent Models (CALM), a method for learning a representation of movement that captures the complexity and diversity of human motion, while also providing a directable interface for controlling a character’s movements. Given raw motion capture recordings, illustrated in Figure 1, CALM encodes them into a latent representation. CALM further decodes a given latent vector into a skill for a physically simulated character, enabling it to perform high-level tasks while being conditioned on a desired motion. As can be seen in Figure 1, without any further training, using only predefined raw motion data, the agent can solve challenging tasks using a set of desired skills."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* CALM creates a direct bidirectional bridge between the raw motion dataset $\mathcal{M}$ and the physics simulator:
  $$\text{Motion Clip } M \xrightarrow{\text{Encoder } E} z \in \mathcal{Z} \xrightarrow{\text{Policy } \pi(a|s,z)} \text{Physics-Simulated Trajectory } \tau$$
* Because $E(M)$ maps any motion clip $M$ into $z$, we obtain an **analytical query mechanism**: any demonstrated clip in the library can be instantly translated into a control latent $z$.
* Downstream tasks can then be controlled in a zero-shot fashion by chaining desired motion clips through their latent codes.

#### 2. Unitree G1 Humanoid Control Relevance
* For G1, this means you can build a library of reference human motions: walking, turning, squatting, reaching, carrying, waving.
* Passing these clips through $E(M)$ gives you the exact latent coordinates $z$ to command the G1 in Isaac Gym or on the physical robot.

---

### Paragraph 1.4 (Distributional Resemblance vs Strict Replication)
> **Original Paper Text:**
> *"A key benefit of our framework is that the policy does not need to precisely replicate the original reference motion. Instead, it has the flexibility to produce diverse movements, as long as they resemble the distributional characteristics of the particular motion clip. This enables the policy to deviate from the motion data and generate new and diverse behaviors that appear natural even though they were not explicitly depicted in the original dataset."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Why Exact Tracking Fails in Robotics:**
  MoCap data comes from human actors whose mass, limb lengths, inertia tensors, and joint limits differ from the simulated character or robot. Strict kinematic tracking forces the robot to mimic human joint trajectories that might violate robot dynamics, leading to tipping over or actuator saturation.
* **Distributional Imitation:**
  CALM's objective matches state-transition distributions $d^\pi(s, s' | z) \approx d^M(\hat{s}, \hat{s}')$ via a conditional discriminator. The policy is rewarded for staying inside the distribution manifold, but has the freedom to adjust foot contact timings, torso lean, and joint torques to maintain physical equilibrium.

#### 2. Unitree G1 Humanoid Control Relevance
* The G1 has specific kinematic differences from human skeletons: ankle motors have limited roll/pitch ranges, the knee joint uses a four-bar or planetary transmission, and total weight is ~35 kg.
* Distributional imitation allows G1 to capture the *essence* of a human movement while adjusting its joint torques and Center of Mass (CoM) trajectory to avoid falling.

---

### Paragraph 1.5 (Unsupervised End-to-End Skill Representation)
> **Original Paper Text:**
> *"Compared to prior work, we focus on unsupervised techniques. The data is unlabeled and we do not assume prior knowledge of semantic connections between motions. We present an end-to-end method that jointly learns a meaningful semantic representation of skills and a control policy capable of producing the selected skills. While previous work used language to derive semantic connections [Juravsky et al. 2022], we demonstrate that semantic meaning can be directly inferred from the motions and similarity is determined in the context of the policy reproducing the motions. This ability to direct the character’s motions then enables tasks to be solved using human-like control and without further re-training, demonstrating CALM’s ability to generate interactive and directable virtual characters."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* Prior directable models like PADL (Juravsky et al. 2022) relied on text annotations (e.g. CLIP embeddings of "walk forward", "punch") to establish semantics. But text annotations are expensive, ambiguous, and low-bandwidth.
* CALM demonstrates that the physical dynamics of the control policy itself naturally induce semantic geometry:
  Motions requiring similar joint dynamics, foot contact patterns, and momentum (e.g. jog, sprint, fast walk) naturally cluster together in the latent space $\mathcal{Z}$ because the policy $\pi(a|s,z)$ reuses similar neural network representations to generate them.

#### 2. Unitree G1 Humanoid Control Relevance
* You do not need to manually label thousands of motion clips in your G1 dataset. The CALM training pipeline automatically discovers skill clusters (locomotion, arm reaching, torso bending) purely through adversarial physical imitation in simulation.

---

### Paragraph 1.6 (Summary of Key Contributions)
> **Original Paper Text:**
> *"To conclude, our contributions are as follows:
> (1) We present a method for jointly training a generative motion controller and a motion encoder, from unlabeled motion capture data. The resulting policy can be directed to generate a motion $M$ via its encoding $z = E(M)$.
> (2) We introduce precision training, a way to reuse the pretrained policy and leverage similarity within the learned latent space to enable control over the produced motion when solving high-level tasks, such as locomotion.
> (3) Finally, we show that combining steps 1 and 2 enables the design of simple FSMs to solve tasks without further training or meticulous design of reward functions or termination conditions."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Contribution 1:** An end-to-end joint training architecture coupling a conditional discriminator $\mathcal{D}(s, s' | z)$, a motion encoder $E(M)$, and a latent-conditioned policy $\pi(a|s, z)$.
* **Contribution 2:** A secondary training phase ("Precision Training") where the low-level controller is frozen, and a high-level policy $\pi^H(z_t | s_t, d^*, \hat{z})$ learns to steer the robot towards a target direction $d^*$ while staying within the neighborhood of a desired motion style $\hat{z}$.
* **Contribution 3:** Zero-shot composition using Finite State Machines (FSMs), avoiding complex multi-objective reward engineering for multi-stage tasks.

#### 2. Unitree G1 Humanoid Control Relevance
* This 3-tier hierarchy maps directly to the Unitree G1 software stack:
  1. **Tier 1 (LLC):** Low-Level Latent Controller runs at 50 Hz on the G1 onboard computer (NVIDIA Jetson / x86 compute unit).
  2. **Tier 2 (HLC):** High-Level Steering Policy runs at 10 Hz, taking joystick commands (heading + speed) and outputting $z_t$.
  3. **Tier 3 (Mission FSM / Behavior Tree):** Autonomous planner switching between navigation, inspection, and manipulation states without re-training RL policies.

---

# PART II: RELATED WORK, RL BACKGROUND & FRAMEWORK OVERVIEW

---

## 2. Related Work

### Paragraph 2.1 (Categorization of Data-Driven Motion Generation)
> **Original Paper Text:**
> *"We aim to learn rich and reusable skill representations, from diverse unlabeled data, for character control. The field of data-driven motion generation can be broadly divided into kinematic models and physics-based models. Kinematic models directly generate pose trajectories without explicitly considering physical constraints, while physics-based models often use a physically simulated environment to enforce realistic dynamics during motion generation."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Kinematic Models:**
  Directly predict joint rotation trajectories $q(t) \in \mathbb{R}^D$ (e.g., via Diffusion models, Motion Matching, or Autoregressive Transformers).
  - *Strengths:* Visually smooth, fast, no physics engine required.
  - *Fatal Flaws for Robotics:* Ignores Newton-Euler equations of motion ($M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q) = \tau + J^T f_c$). Kinematic trajectories suffer from foot floating, ground interpenetration, and impossible accelerations that real motors cannot generate.
* **Physics-Based Models:**
  The neural network acts inside a physics simulator (e.g. Isaac Gym, MuJoCo). It outputs control actions (such as joint target angles $q_{\text{target}}$ to Proportional-Derivative (PD) controllers or torques $\tau$). The simulator integrates the rigid-body equations forward in time, strictly enforcing contact dynamics, gravity, friction, and joint limits.

#### 2. Unitree G1 Humanoid Control Relevance
* On the Unitree G1 robot, kinematic models *cannot* be run directly on the hardware: commanding kinematic joint angles without accounting for contact forces and balance causes the bipedal robot to immediately topple over.
* Physics-based modeling is mandatory: the policy must learn to manage ground reaction forces (GRFs) through the G1's foot soles to stay upright.

---

### Paragraph 2.2 (Control Modalities: Direct Prediction vs Latent-Based Control)
> **Original Paper Text:**
> *"We differentiate between two methods of control for parametric models: direct prediction and latent/language-based control. In direct prediction, the model learns to generate motions for a predefined task, while latent and language-based control involves learning a generative model for motions. As our focus is on creating interactive, controllable characters, we build upon the literature of physics-constrained generation and focus on latent-based control."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Direct Prediction:**
  Trains an RL policy directly to maximize a specific downstream task reward (e.g., $r = v_x - |v_y|$).
  - *Drawback:* Task-specific policies are throwaway. If the task changes from walking to climbing or carrying a load, the entire network must be retrained from scratch.
* **Latent-Based Generative Control:**
  Decouples motor competence from task planning. A low-level policy is trained once as a universal motion generator conditioned on a continuous latent vector $z \in \mathcal{Z}$. Any downstream task is then solved by searching or learning in the low-dimensional latent space $\mathcal{Z}$ rather than the high-dimensional motor action space $\mathcal{A}$.

#### 2. Unitree G1 Humanoid Control Relevance
* For G1, direct RL for every single task (walk, turn, carry, push, open door) is prohibitively sample-inefficient and dangerous to deploy.
* With latent-based control, the G1 acquires a foundational "motor brain" (locomotion, arm balance, stepping strategies) that remains permanently fixed, while lightweight high-level task planners simply modulate $z$.

---

### Paragraph 2.3 (Subsection 2.1: Physics-Constrained Motion Generation & DeepMimic)
> **Original Paper Text:**
> *"In physics-constrained generation, the model predicts motor actuations. These are fed into the simulator, which then produces the next state by emulating the character’s motion while adhering to the various governing laws (such as gravity and friction). Recent advances in deep learning, specifically deep reinforcement learning, have enabled a leap forward in generation quality. Such an example is DeepMimic [Peng et al. 2018], in which they defined a motion-tracking reward that is used for training a policy to imitate specific motions. However, these schemes focused on imitating single, pre-defined, motion clips."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* DeepMimic demonstrated that Deep RL with PPO can produce acrobatic skills (spinkicks, cartwheels) in simulated humanoids.
* The tracking reward strictly enforces matching a time-indexed reference pose $\hat{q}(t)$:
  $$r_t^{\text{pose}} = \exp\left(-\sum_j \| q_j(t) \ominus \hat{q}_j(t) \|^2\right)$$
* *The Core Limitation:*
  1. The policy requires a phase variable $\phi \in [0, 1)$ indicating progress through the clip.
  2. It cannot transition dynamically between different clips without manually crafted blend trees.
  3. Scaling to a library of 200 motions would require 200 distinct policies or massive networks with separate tracking phase clocks.

#### 2. Unitree G1 Humanoid Control Relevance
* Retargeting human MoCap to G1 via DeepMimic is brittle: if the G1 trips or hits an unexpected obstacle, the phase clock $\phi$ keeps ticking forward while the physical robot is stuck on the ground, causing catastrophic joint thrashing.
* G1 requires a phase-free, reactive motor prior like CALM.

---

### Paragraph 2.4 (Direct Prediction: Adversarial Motion Priors - AMP)
> **Original Paper Text:**
> *"Direct prediction. In direct prediction, the goal is to directly learn to solve a downstream task. For instance, the agent may be tasked with reaching a goal location. Here, AMP [Peng et al. 2021] combines adversarial imitation learning [Ho and Ermon 2016, GAIL] with classic RL. The agent attempts to balance between maximizing the task reward, whilst successfully fooling a discriminator. However, when the demonstration data distribution does not fit the task, the resulting behavior is unsatisfactory, and the resulting model is unable to generalize to new tasks – requiring re-optimizing the provided data and re-training per each task."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **AMP Objective:**
  AMP formulates the reward as:
  $$r_t = w_{\text{task}} r_t^{\text{task}} + w_{\text{style}} r_t^{\text{style}}(s_t, s_{t+1})$$
  where $r_t^{\text{style}} = -\log(1 - \mathcal{D}(s_t, s_{t+1}))$.
* **The Failure Mode of AMP:**
  1. *Mode Collapse / Selection Bias:* If the dataset contains walking, jogging, crawling, and backflipping, but the task is "move forward at 1 m/s", the agent discovers that "jogging" easily satisfies the task reward and satisfies the discriminator. It completely discards the rest of the dataset.
  2. *Data-Task Conflict:* If the task requires moving backwards or crawling through a tunnel, but the MoCap dataset only contains upright forward walking, the discriminator heavily penalizes crawling. The policy gets stuck between two competing gradients (task reward vs style discriminator).
  3. *No Reusability:* Every new task requires re-running expensive multi-billion-step adversarial RL.

#### 2. Unitree G1 Humanoid Control Relevance
* On the Unitree G1, running AMP means if you want G1 to walk carrying a box, you must train an AMP policy from scratch with box-carrying MoCap. If you next want G1 to push a cart, you must retrain another AMP policy. This is impractical for real-world robotics development.

---

### Paragraph 2.5 (Latent-Based Control: Park et al., Won et al., and ASE)
> **Original Paper Text:**
> *"Latent-based control. Here the task is to learn a behavior manifold that can be sampled from. Park et al. [2019] learn to predict future states and then a controller to track that behavior. Won et al. [2022] learn a conditional variational auto-encoder (VAE), conditioned on the next state. Finally, closest to our work, ASE [Peng et al. 2022] presents an unsupervised discriminative learning procedure. By maximizing the mutual information between the latent space and the produced next state, in addition to optimizing the imitation learning objective, the agent learns to generate diverse motions. These methods share a common theme – the resulting latent space is complex to control. In this work, we learn a dense representation of human motion jointly with a directable latent-conditioned policy."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **ASE (Adversarial Skill Embeddings):**
  Uses the Information Maximization (InfoMax) principle (similar to DIAYN - Diversity is All You Need):
  $$\max_\pi \mathcal{H}(S') - \mathcal{H}(S' | Z) = I(S'; Z)$$
  In practice, this is optimized by training an encoder/decoder network $q_\psi(z | s')$ to predict which latent $z$ caused the transition $s'$.
* **Why ASE Latent Space Fails Controllability:**
  Because $z$ is sampled from a prior (e.g. uniform on a hypersphere) without any reference to which motion clip it corresponds to, the latent space is a black box. If you want the character to kick with its right foot, you have no analytical way of knowing what latent vector $z$ will produce a right-foot kick. You have to train a complex high-level policy to search the latent space via trial-and-error RL.
* **CALM's Solution:**
  CALM binds each latent $z$ directly to a reference motion $M$ through an explicit encoder $z = E(M)$.

#### 2. Unitree G1 Humanoid Control Relevance
* With ASE, commanding the G1 requires training a high-level policy for every single action, with no guarantee that the desired gait exists in the learned manifold.
* With CALM, G1's operator can record a 2-second motion clip (e.g. human stepping over an obstacle), pass it through $E(M)$, and immediately execute that skill on the robot.

---

### Paragraph 2.6 (Subsection 2.2: Representation Learning - VAEs vs Adversarial Methods)
> **Original Paper Text:**
> *"Representation learning, the task of capturing the underlying structure of data, has been a significant area of focus in the field of machine learning, particularly in the representation of images and videos. There are different ways to define similarity between data points, but one approach that is closest to our method is by directly utilizing a downstream task. For instance, generative adversarial networks [Goodfellow et al. 2020, GAN] and VAEs [Kingma and Welling 2013] learn a low-dimensional latent representation in order to recreate data from the reference distribution.
> In the context of learning representations for motions, VAE and adversarial-based training can be differentiated. Motion VAEs [Ling et al. 2020; Won et al. 2022] enable jointly learning to represent and generate motion. While VAE-based methods are typically easier to train, their pitfall is that they are driven by a reconstruction loss, preventing the ability to deviate from the data distribution."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Motion VAEs:**
  Optimize the Evidence Lower Bound (ELBO):
  $$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) \parallel p(z))$$
  The reconstruction term $\log p(x|z)$ penalizes the L2 distance between the simulated joint positions and the MoCap poses:
  $$\mathcal{L}_{\text{recon}} = \| s_{\text{sim}} - s_{\text{mocap}} \|^2$$
* **The Pitfall of Reconstruction Loss for Dynamic Agents:**
  When a physics-based character experiences a dynamic collision or an uneven step, it cannot reproduce the exact kinematics of the MoCap recording. If forced by an L2 loss, the policy incurs massive penalties and often collapses or falls ("the sinking problem"). It cannot deviate gracefully to regain balance.

#### 2. Unitree G1 Humanoid Control Relevance
* On a physical Unitree G1, foot placement will never match a human MoCap recording down to the millimeter due to gear backlash, joint compliance, and ground stiffness.
* A reconstruction loss penalizes the G1 for taking an extra corrective step to prevent a fall. An adversarial distribution loss, by contrast, rewards the G1 for staying upright and maintaining a human-like stepping pattern.

---

### Paragraph 2.7 (Adversarial Generalization & Skill Transitions)
> **Original Paper Text:**
> *"The benefit of adversarial methods is in their ability to generate new motions that are likely under the reference data distribution. This is important in the context of controllable characters. The character must transition naturally between every motion pair, even when it is not provided with an explicit demonstration."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* Adversarial imitation learning (GAIL / AMP / CALM) trains a discriminator $\mathcal{D}$ to classify whether a transition $(s, s')$ belongs to the distribution of human motions $p_{\text{data}}(s, s')$.
* If the agent is currently sprinting and receives a command to transition to crouch-idle, there is no single MoCap clip in the dataset that depicts this exact transition.
* An adversarial policy interpolates smoothly through the state space, generating intermediate states that look natural and physically plausible under the discriminator, even though they were never explicitly recorded in the MoCap dataset.

#### 2. Unitree G1 Humanoid Control Relevance
* Real-world humanoid control is 90% about **transitions**: transitioning from forward walking to stopping, from standing to squatting, from balancing on one foot to recovering. A dataset cannot contain every pairwise transition ($N^2$ combinations). CALM's adversarial formulation ensures G1 smoothly and stably navigates between skills without falling.

---

### Paragraph 2.8 (Limitations of ASE and Language Supervision / PADL)
> **Original Paper Text:**
> *"As shown in ASE [Peng et al. 2022], the policy learns to generate diverse behaviors, generalizing beyond motion reconstruction. However, their resulting latent space lacks global semantic structure. As a result, ASE tends to mode-collapse and does not provide an easy way to map motions to the latent space, an important requirement for directability and control. Alternative work such as PADL [Juravsky et al. 2022] utilized supervision from natural language. These methods are able to train language-aligned motion representations that can then generate behaviors based on natural language commands. However, they require access to labeled data. In this work, the data is unlabeled and the representation is learned end-to-end with the imitation learning policy. Hence, the semantic connections between motions are determined in the context of the policy reproducing the motions."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **PADL (Language-Conditioned Skills):** Uses CLIP embeddings of text descriptions ("walk", "swing sword") as condition vectors.
  - *Limitation:* Natural language lacks spatial precision. You cannot describe the exact nuance, arm height, or stride length of a complex motion in a single text prompt. Text datasets are also expensive and human-subjective.
* **CALM's Unsupervised Semantic Grounding:**
  CALM requires zero text labels. Semantic connections arise purely because similar motions share similar dynamical state-transitions $(s, s')$. The encoder $E(M)$ groups motions with similar kinematic patterns, and the policy $\pi(a|s, z)$ enforces that latent proximity corresponds to behavioral similarity.

#### 2. Unitree G1 Humanoid Control Relevance
* For G1, you can ingest unannotated public MoCap libraries (hundreds of gigabytes of raw optical marker data) directly without hiring humans to annotate every 2-second snippet.

---

### Paragraph 2.9 (Subsection 2.3: Hierarchical Reinforcement Learning & Options Framework)
> **Original Paper Text:**
> *"Latent generative models can be seen as part of the options/skills framework [Sutton et al. 1999]. The options framework differentiates between a high and low-level controller. The low-level controller produces micro-actions, which are high-frequency actions capable of generating diverse behaviors. On the other hand, the high-level controller often plans at a lower frequency. At each time step, it selects which skill to play. Skills are temporally extended actions, or, in our context, long-term motions."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **The Options Framework (Sutton, Precup, Singh 1999):**
  Defines an option $\omega = \langle \mathcal{I}_\omega, \pi_\omega, \beta_\omega \rangle$, where $\mathcal{I}_\omega$ is the initiation set, $\pi_\omega(a|s)$ is the option policy, and $\beta_\omega(s)$ is the termination condition.
* **Continuous Latent Skills as Universal Options:**
  CALM replaces discrete options with a continuous latent manifold $\mathcal{Z}$.
  - Low-level policy $\pi(a_t | s_t, z)$ runs at high frequency ($f_{\text{low}} = 30\text{ Hz}$ in the paper; $50\text{ Hz}$ on G1).
  - High-level policy $\pi^H(z_k | s_k, g)$ runs at lower frequency ($f_{\text{high}} = 6\text{ Hz}$ in the paper; $10\text{ Hz}$ on G1), holding $z$ constant for multiple low-level control steps.
* **Temporal Abstraction Benefits:**
  The high-level policy does not need to worry about individual joint torques, ankle stabilization, or contact timings. Its effective time horizon is reduced by a factor of 5 to 10, drastically simplifying credit assignment in RL.

#### 2. Unitree G1 Humanoid Control Relevance
* On the Unitree G1:
  - Motor PD loops run at 500 Hz – 1 kHz on the motor drive boards.
  - Low-level CALM policy runs at 50 Hz on the onboard CPU/GPU, outputting PD position targets.
  - High-level navigation / manipulation planner runs at 5–10 Hz, outputting skill latents $z_t$.
  - This hierarchical decoupling is essential for real-time compute budgeting on embedded hardware.

---

### Paragraph 2.10 (HRL Restricting Solutions to Data Support)
> **Original Paper Text:**
> *"Prior efforts in hierarchical reinforcement learning (HRL) have shown that utilizing meaningful skills can not only speed up training [Tessler et al. 2017], but also enforce the solution to reside within the support of the data [Peng et al. 2022]."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* In unconstrained RL, the policy explores the entire action space $\mathcal{A} = [-1, 1]^D$. On a humanoid, 99.99% of this space consists of falling down, hyper-extending joints, or violently vibrating.
* By restricting the high-level policy's action space to the latent space $\mathcal{Z}$, every single latent $z \in \mathcal{Z}$ decodes into a physically balanced, human-like motion.
* The exploration space is effectively constrained to the **safe, stable manifold of human locomotion**.

#### 2. Unitree G1 Humanoid Control Relevance
* Deploying unconstrained RL on a real G1 causes catastrophic hardware failures (stripped gears, burnt motor windings, broken structural links from violent falls).
* Constraining the high-level policy to the CALM latent manifold ensures that every commanded action is dynamically stable and mechanically safe for the G1's actuators.

---

## 3. Reinforcement Learning Background

### Paragraph 3.1 (MDP Formulation & Cumulative Return)
> **Original Paper Text:**
> *"In this work, both the pre-training and the downstream tasks are modeled as reinforcement learning problems, where an agent interacts with an environment according to a policy $\pi$. At each step $t$, the agent observes a state $s_t$ and samples an action $a_t$ from the policy $a_t \sim \pi(a_t | s_t)$. The environment then transitions to the next state $s_{t+1}$ based on the transition probability $p(s_{t+1} | s_t, a_t)$. The goal is to maximize the discounted cumulative reward, defined as
> $$J = \mathbb{E}_{p(\tau | \pi)} \left[ \sum_{t=0}^T \gamma^t r_t \Big| s_0 = s \right], \quad (1)$$
> where $p(\tau | \pi) = p(s_0) \prod_{t=0}^{T-1} p(s_{t+1} | s_t, a_t) \pi(a_t | s_t)$ is the likelihood of a trajectory $\tau = [s_0, a_0, r_0, \dots, s_T]$, and $\gamma \in [0, 1)$ is the discount factor that determines whether the agent is short-sighted or considers longer-term outcomes."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* Standard infinite-horizon or finite-horizon Markov Decision Process (MDP) specified by tuple $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, r, \gamma, \rho_0 \rangle$.
* In Section 5, the reward $r_t$ is not a hand-crafted heuristic, but an adversarial signal derived from the conditional discriminator: $r_t = -\log(1 - \mathcal{D}(s_t, s_{t+1} | z))$.
* In Section 6, the reward $r_t$ becomes a task reward combined with a latent style preservation penalty.

#### 2. Unitree G1 Humanoid Control Relevance
* For G1:
  - State $s_t \in \mathbb{R}^{S}$: includes base height, orientation (quaternion / rotation matrix), linear and angular velocities, joint positions $q$, joint velocities $\dot{q}$, previous actions $a_{t-1}$, and contact sensor flags.
  - Action $a_t \in \mathbb{R}^{A}$: target joint position offsets $\Delta q$ added to default standing pose $q_0$, fed to low-level PD controller $\tau = K_p (q_0 + a_t - q) - K_d \dot{q}$.
  - Discount factor $\gamma = 0.99$ balances immediate balance recovery with sustained multi-step gait execution.

---

## 4. Overview

### Paragraph 4.1 (The Three Phases of CALM)
> **Original Paper Text:**
> *"In this paper, we introduce Conditional Adversarial Latent Models (CALM), a scalable, data-driven approach for creating directable controllers for physically simulated characters. These characters can be controlled in a similar way to how players control virtual characters in games, by providing a sequence of instructions for movement and actions. Figure 2 outlines our framework. It consists of three steps. (1) Low-level training of a motion encoder and motion generator. (2) Precision training using a high-level policy. (3) Inference, during which tasks are solved without further training."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* The CALM methodology splits the learning process into three explicit, non-overlapping sequential stages:
  1. **Phase 1: Pre-training (Low-Level Policy + Encoder + Discriminator):** Learns a universal motion generator and latent space $\mathcal{Z}$ from unannotated MoCap.
  2. **Phase 2: Precision Training (High-Level Task Policy):** Teaches a steering controller to direct velocity and heading while preserving motion style $\hat{z}$.
  3. **Phase 3: Inference / FSM Execution:** Assembles complex behaviors zero-shot by chaining high-level steering commands and low-level discrete actions using state machines.

#### 2. Unitree G1 Humanoid Control Relevance
* This 3-phase modular architecture is ideal for G1 development:
  - You train Phase 1 once on an A100 GPU cluster with thousands of Isaac Gym environments.
  - Once Phase 1 is frozen, you never have to re-train the discriminator or the low-level policy.
  - New robot behaviors (e.g. patrol, follow human, carry object) are implemented in Phase 2 or Phase 3 in minutes rather than days.

---

### Paragraph 4.2 (Phase 1: Motion Encoder and Low-Level Decoder)
> **Original Paper Text:**
> *"During low-level training, CALM learns an encoder $E$. It takes a motion $M$ from a reference dataset of motions $\mathcal{M}$, a time-series of joint locations, and maps it into a low-dimensional latent representation $z \in \mathcal{Z}$. Additionally, CALM also jointly learns a decoder. The decoder is a low-level policy $\pi(a|s, z)$ that interacts with the simulator and generates motions similar to the reference dataset. This policy produces a variety of behaviors on demand, but is not conditioned on the directionality of the motion. For example, it can be instructed to walk, but does not enable intuitive control over the direction of walking."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Input to Encoder $E(M)$:** A 2-second motion clip $M = \{ \hat{x}_j(t) \}_{t=1}^H$, represented as 3D Cartesian coordinates of joints in the character's root-relative coordinate frame.
* **Latent Representation $z$:** Projected onto the 64-dimensional unit hypersphere $\mathbb{S}^{63}$.
* **The Low-Level Policy $\pi(a|s, z)$:** Acts as a decoder. Given current physical state $s_t$ and latent $z$, it generates actions $a_t$ that cause the character to physically exhibit the motion $M$.
* **The Directionality Limitation:**
  Notice this key insight: The low-level policy reproduces the *intrinsic gait* of the clip (e.g. walking, turning in circles, or crouching), but if the original clip was walking along heading $\theta = 45^\circ$, the policy will walk along heading $45^\circ$ relative to its starting orientation. It does not provide an external steering wheel to guide the robot towards arbitrary target locations $(x^*, y^*)$. That is delegated to Phase 2.

#### 2. Unitree G1 Humanoid Control Relevance
* When training G1 on MoCap, human actors in motion capture suits walk in arbitrary directions within the capture volume. The low-level policy simply learns the physics of bipedal stepping for each style. Steering the G1 towards a specific door or waypoint is handled by the high-level policy.

---

### Paragraph 4.3 (Phase 2: Precision Training & Directional Steering)
> **Original Paper Text:**
> *"Next, to control motion direction, we train a high-level task-driven policy to select latent variables $z_t$. These latents are provided to the low-level policy which generates the requested motion. Here, the motion encoder is used to constrain the latents $z_t$ to be close to pre-specified motions $\hat{z}_t$, thus guiding the high-level policy to adopt a desired behavioral style. For example, move in a given direction while performing a crouch-walking motion."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* The high-level policy $\pi^H(z_t | s_t, d^*, \hat{z})$ outputs latents $z_t \in \mathcal{Z}$.
* The input includes:
  - Current physical state $s_t$.
  - Target heading direction $d^* \in \mathbb{R}^2$ (unit vector in root local coordinates).
  - Desired style latent $\hat{z} = E(M_{\text{style}})$.
* The reward balances two terms:
  1. *Task / Steering Reward:* Encourages linear velocity $\dot{x}_{\text{root}}$ to align with $d^*$.
  2. *Latent Style Constraint:* Penalizes $\| z_t - \hat{z} \|^2$, forcing the high-level policy to search only within the local latent neighborhood of the requested style.

#### 2. Unitree G1 Humanoid Control Relevance
* For G1, this allows operators to command: *"Walk towards $(x^*, y^*)$ in a crouched stealth gait"* or *"Sprint towards $(x^*, y^*)$"*.
* The high-level policy modulates $z_t$ slightly away from $\hat{z}$ (e.g. leaning into turns, lengthening stride) to satisfy the heading command while strictly preserving the requested crouch-walking or sprinting gait.

---

### Paragraph 4.4 (Phase 3: Zero-Shot Task Solution via Finite State Machines)
> **Original Paper Text:**
> *"Finally, the previously trained models are combined to compose complex movements without additional training. To do so, the user produces a finite-state machine (FSM) containing standard rules and commands. These determine which motion to perform, similar to how a user controls a video game character. For example, they determine whether the character should perform a simple motion, performed directly using the low-level policy, or a directed motion requiring high-level control. As an example, one may construct an FSM like (a) 'crouch-walk towards the target, until distance < 1m', then (b) 'kick', and finally (c) 'celebrate' (Figure 1)."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* In standard Deep RL, multi-stage tasks (navigate to object $\to$ manipulate object $\to$ return to base) are notoriously difficult to train using monolithic reward functions due to reward balancing, exploration bottlenecks, and catastrophic forgetting.
* In CALM, the multi-stage task is decomposed into an FSM:
  - *State 1 (Directed Locomotion):* FSM sends target direction $d^*$ and style $\hat{z}_{\text{crouch}}$ to High-Level Policy $\pi^H$.
  - *State Transition:* Distance to target $< 1.0\text{ m}$.
  - *State 2 (Discrete Action):* FSM feeds $\hat{z}_{\text{kick}} = E(M_{\text{kick}})$ directly to Low-Level Policy $\pi$.
  - *State Transition:* Kick animation completed (after $T_{\text{kick}}$ steps).
  - *State 3 (Discrete Action):* FSM feeds $\hat{z}_{\text{celebrate}} = E(M_{\text{celebrate}})$ directly to Low-Level Policy $\pi$.
* Zero training is required for this complex composite mission!

#### 2. Unitree G1 Humanoid Control Relevance
* This directly mirrors the autonomy architecture of industrial humanoid robots.
* A mission planner (ROS 2 BehaviorTree / FSM) controls the G1:
  - State 1: Walk to inspection station (via High-Level Steering Policy).
  - State 2: Reach out arm to inspect barcode (via Low-Level Policy conditioned on reaching MoCap).
  - State 3: Turn around and walk to pallet (via High-Level Steering Policy).
* No reward engineering or RL training is needed for each custom warehouse workflow!

---

# PART III: CONDITIONAL ADVERSARIAL LATENT MODELS (METHOD) & HIGH-LEVEL CONTROL

---

## 5. Conditional Adversarial Latent Models

### Paragraph 5.1 (Core Problem Formulation & Objective)
> **Original Paper Text:**
> *"Learning a rich reusable skill representation enables characters to generate a wide variety of motions on demand, opening a range of potential applications such as games and visual effects. In CALM, these skills are modeled using a motion-conditioned policy $\pi(a|s, z)$, where motions $M$ are encoded into a latent variable $z \in \mathcal{Z}$. This mapping is modeled using a motion encoder $z = E(M)$, and learned by solving a conditional imitation learning objective over motions $M$ sampled uniformly from a reference dataset $\mathcal{M}$:
> $$\max_\pi -\mathbb{E}_{M \in \mathcal{M}} \left[ D_{\text{JS}} \left( d^\pi(s, s' | z)\big|_{z=E(M)} \parallel d^M(\hat{s}, \hat{s}') \right) \right], \quad (2)$$
> where $D_{\text{JS}}$ is the Jensen-Shannon divergence, and $d^\pi(s, s' | z)|_{z=E(M)}$ and $d^M(\hat{s}, \hat{s}')$ are respectively the state transition distribution of the policy and reference motions."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Mathematical Anatomy of Equation (2):**
  - $\mathcal{M}$: Dataset of reference motion capture clips.
  - $M$: A specific motion clip sampled uniformly from $\mathcal{M}$.
  - $z = E(M)$: Low-dimensional latent representation output by the encoder.
  - $d^M(\hat{s}, \hat{s}')$: The empirical distribution of consecutive state transitions in motion clip $M$.
  - $d^\pi(s, s' | z)$: The distribution of state transitions generated by the physics policy $\pi$ when conditioned on latent $z$.
  - $D_{\text{JS}}(P \parallel Q)$: The Jensen-Shannon divergence:
    $$D_{\text{JS}}(P \parallel Q) = \frac{1}{2} D_{\text{KL}}\left(P \parallel \frac{P+Q}{2}\right) + \frac{1}{2} D_{\text{KL}}\left(Q \parallel \frac{P+Q}{2}\right)$$
* **Why Jensen-Shannon Divergence?**
  Unlike Kullback-Leibler (KL) divergence, JS divergence is symmetric, bounded ($0 \le D_{\text{JS}} \le \log 2$), and finite even when the distributions $P$ and $Q$ have disjoint supports (which is common in high-dimensional physics trajectories).
* **The Conditional Essence:**
  The critical difference from standard GANs / AMP is conditioning on $z = E(M)$. The policy is not tasked with producing *any* valid human motion; it must minimize divergence specifically against the transition distribution of clip $M$.

#### 2. Unitree G1 Humanoid Control Relevance
* On the Unitree G1, $d^M$ represents retargeted MoCap transitions (joint angles, velocities, end-effector poses) of a specific human clip (e.g. sidestepping).
* Equation (2) dictates that when G1 is fed $z = E(M_{\text{sidestep}})$, its simulated physical transitions must match the distribution of sidestepping, preventing the robot from substituting an easier forward step.

---

### Paragraph 5.2 (Imitation from Observation - GAIfO & Mode Collapse)
> **Original Paper Text:**
> *"This objective is related to prior work in imitation learning literature, namely learning from observations (LfO) [Torabi et al. 2018]. In this setup, the agent is provided with a demonstration dataset that consists only of state transitions/observations, without the underlying actions that produced those transitions. Generative Adversarial Imitation from Observation [Torabi et al. 2018, GAIfO] jointly trains a policy and a discriminator. The policy generates state transitions $(s, s')$, while the discriminator tries to distinguish them from demonstrations sampled from the data $(\hat{s}, \hat{s}')$. However, provided a variety of motions, GAIfO is prone to mode-collapse, where the policy does not model all of the different behaviors from a large dataset, and the resulting model does not explicitly learn a skill embedding that can be used to direct the policy on downstream tasks."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Why Learning from Observations (LfO)?**
  Optical motion capture systems record 3D joint positions and orientations $(\hat{s}, \hat{s}')$. They *never* record the ground-truth motor torques or muscle activations $(\hat{a})$ that produced those movements. Thus, standard Behavioral Cloning (BC), which requires $(s, a)$ pairs, is impossible.
* **GAIfO Formulation:**
  GAIfO solves this by training a discriminator $\mathcal{D}(s, s')$ to distinguish real transitions $(\hat{s}, \hat{s}')$ from simulated transitions $(s, s')$.
* **The Mode Collapse Failure Mode:**
  When $\mathcal{M}$ contains 160 distinct motions, the policy discovers that performing a generic "walking" motion easily satisfies $\mathcal{D}$ across many states. The policy completely ignores hard motions (e.g. low squats, kicks, sudden spins) because it is evaluated on aggregate data likelihood. Furthermore, GAIfO provides no latent conditioning vector to selectively command specific motions.

#### 2. Unitree G1 Humanoid Control Relevance
* Human MoCap datasets (AMASS, CMU) provide human kinematics, not G1 motor torques. LfO is therefore required.
* But vanilla GAIfO would result in a G1 that only learns a bland forward shuffle and loses the ability to crouch, jump, or execute diverse manipulation stances.

---

### Paragraph 5.3 (The ASE Critique: Mutual Information vs Directability)
> **Original Paper Text:**
> *"In previous research, ASE [Peng et al. 2022] attempted to address these problems by introducing a latent variable and maximizing mutual information. They demonstrated that the model can produce diverse behavior when provided with randomly sampled variables. However, solely relying on mutual information loss is not enough. To avoid mode-collapse, ASE also employs a diversity loss, which states that similar latent variables should produce similar action distributions. Our study shows that this is particularly important in terms of directability. The encoder learns an imprecise mapping between motions and latents, resulting in a controller that does not produce similar motions when conditioned on the same latent, making it challenging to direct the policy to perform specific motions."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **ASE's Mechanism:**
  ASE samples random latent codes $z \sim \text{Uniform}(\mathbb{S}^{d-1})$ and optimizes:
  $$\max_\pi I(S'; Z) = \mathbb{E}_{s, z, s'} [\log q_\psi(z | s') - \log p(z)]$$
  alongside an unconditioned discriminator $\mathcal{D}(s, s')$.
* **Why Mutual Information Fails Directability:**
  1. *Unpaired Latent Space:* The latent $z$ has no semantic anchor to any real-world motion. $z$ is merely an abstract random index.
  2. *Inversion Problem:* If you have a motion clip $M$ and want the policy to reproduce it, you must train an auxiliary decoder to guess what $z$ might produce $M$.
  3. *High Variance:* Because $q_\psi(z | s')$ tries to infer $z$ from a single transition $s'$, different runs map similar motions to wildly different regions of the latent sphere. Controllability is compromised.

#### 2. Unitree G1 Humanoid Control Relevance
* For G1 robotics engineers, ASE is unpredictable. If you command latent vector $z$, you cannot know with mathematical confidence whether the G1 will perform a stable step or an erratic lunge. CALM's direct mapping $z = E(M)$ provides deterministic, reproducible motor execution.

---

### Paragraph 5.4 (The CALM Solution: Conditional Discriminator & Policy Reward)
> **Original Paper Text:**
> *"CALM mitigates the issues from prior methods by using a conditional discriminator that forces the policy to reproduce each motion in the dataset. At each iteration, a random motion $M$ is sampled from the reference dataset. The encoder maps the motion to a latent encoding $z = E(M)$. Both the policy and the conditional discriminator are then conditioned on this latent $z$. Conditioning the discriminator on the latent helps to mitigate mode collapse, by forcing the policy to produce motions that are similar to the corresponding motion of a given latent. This leads to the following discriminator loss
> $$\mathcal{L}_D = -\mathbb{E}_{M \in \mathcal{M}} \left[ \mathbb{E}_{d^M(\hat{s}, \hat{s}')} [\log \mathcal{D}(\hat{s}, \hat{s}' | z)] + \mathbb{E}_{d^\pi(z)(s, s')} [\log(1 - \mathcal{D}(s, s' | z))] \Big| z = E(M) \right], \quad (3)$$
> and policy objective
> $$J = \mathbb{E}_{M \in \mathcal{M}} \left[ \mathbb{E}_{p(\tau | \pi, z)} \left[ \sum_t \gamma^t r(s_t, s_{t+1}, z) \Big| s_0 = s \right] \Big| z = E(M) \right], \quad (4)$$
> with rewards $r(s_t, s_{t+1}, z) = -\log(1 - \mathcal{D}(s_t, s_{t+1} | z))$."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Conditioning the Discriminator $\mathcal{D}(s, s' | z)$:**
  This is the mathematical core of CALM.
  - In standard AMP/GAIL, the discriminator asks: *"Is $(s, s')$ a human motion?"*
  - In CALM, the conditional discriminator asks: *"Is $(s, s')$ the specific motion encoded by $z = E(M)$?"*
* **Mathematical Equilibrium:**
  When $\mathcal{D}$ reaches optimal discriminator status $\mathcal{D}^*(s, s' | z) = \frac{d^M(s, s')}{d^M(s, s') + d^\pi(s, s' | z)}$, the policy objective in Equation (4) exactly minimizes the Jensen-Shannon divergence $D_{\text{JS}}(d^\pi(s, s' | z) \parallel d^M(s, s'))$.
* **Why Mode Collapse is Eradicated:**
  If the policy generates a generic walk when $z = E(M_{\text{crouch}})$, the discriminator $\mathcal{D}(s_{\text{walk}}, s'_{\text{walk}} | z_{\text{crouch}})$ immediately detects the mismatch and outputs near-zero probability. The policy receives a massive penalty ($r \to -\infty$), forcing it to replicate the true crouch.

#### 2. Unitree G1 Humanoid Control Relevance
* On the G1, this ensures 100% skill coverage across the entire dataset. If your MoCap library has 50 diverse locomotion styles and 30 manipulation stances, the G1 low-level policy is mathematically forced to master every single one of them.

---

### Paragraph 5.5 (Subsection 5.1: Practical Considerations)
> **Original Paper Text:**
> *"In this section, we present design decisions needed for a practical instantiation of CALM, as well as in-depth implementation details."*

---

### Paragraph 5.6 (Subsubsection 5.1.1: Encoder Architecture & End-to-End Training)
> **Original Paper Text:**
> *"An ideal representation is one that is optimized for the control policy, rather than relying on auxiliary objectives to drive the structure of the latent representation, as in contrastive learning methods [Oord et al. 2018]. We show that an effective encoder can be trained in an end-to-end fashion using gradients from the policy when optimizing the pre-training objective Equation (4). Furthermore, this approach results in a latent space with a clear semantic structure, where similar motions are grouped closely, enabling interpolation in a semantically meaningful manner."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **End-to-End Gradients:**
  Instead of pre-training the encoder with an isolated self-supervised loss (e.g., Contrastive Predictive Coding / SimCLR) and freezing it, CALM backpropagates policy performance gradients directly into $E(M)$:
  $$\nabla_\theta J(\theta, \phi) = \mathbb{E} \left[ \nabla_z Q(s, a, z) \cdot \nabla_\phi E_\phi(M) \right]$$
* **Why This Matters:**
  The encoder is forced to organize the latent space according to **controllability**. If two motions require completely different torque profiles and contact transitions, the policy gradient forces their latent codes apart. If two motions share physical dynamics, they are drawn together.

#### 2. Unitree G1 Humanoid Control Relevance
* For G1, this means the latent space reflects the robot's **actuator physics**, not just visual geometry. Movements that have similar torque demands on G1's hip and knee actuators naturally become neighbors in latent space.

---

### Paragraph 5.7 (Latent Space Geometry: $l_2$ Hypersphere Projection)
> **Original Paper Text:**
> *"The encoder’s output is projected onto the $l_2$ unit hypersphere. This constraint is inspired by prior work in the field [Bojanowski and Joulin 2017; Chen et al. 2020; Parkhi et al. 2015; Peng et al. 2022; Wang et al. 2017; Wang and Isola 2020; Xu and Durrett 2018], and has several benefits. For example, fixed-norm vectors are known to improve training stability in machine learning, where dot products are commonly used [Wang et al. 2017; Xu and Durrett 2018]. In the context of motion generation, the structure imposed on the latent space by the unit $l_2$ norm constraint reduces the likelihood of unnatural behaviors arising from sampling out-of-distribution latents during inference, as demonstrated by Peng et al. [2022]."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Mathematical Projection:**
  $$z = \frac{E(M)}{\| E(M) \|_2}, \quad z \in \mathbb{S}^{d-1}$$
  where $d = 64$.
* **Geometric & Optimization Advantages:**
  1. *Bounded Support:* Unbounded Euclidean spaces $\mathbb{R}^d$ suffer from "holes" and scale drift. The policy might encounter vectors with magnitude $\|z\| = 10.0$ during inference, causing extreme out-of-distribution actions. On $\mathbb{S}^{d-1}$, every point has $\|z\| = 1$.
  2. *Uniform Interpolation:* Linear or spherical linear interpolation (SLERP) along great arcs on $\mathbb{S}^{d-1}$ is smooth and well-behaved:
     $$z(t) = \frac{\sin((1-t)\theta)}{\sin\theta} z_1 + \frac{\sin(t\theta)}{\sin\theta} z_2$$

#### 2. Unitree G1 Humanoid Control Relevance
* On the physical G1 robot, feeding an out-of-distribution latent to the low-level controller could command sudden 180-degree joint snap, damaging planetary gears. The unit hypersphere bounds the operational domain, ensuring safe bounded inputs.

---

### Paragraph 5.8 (Temporal Sub-Sequences & Latent Regularization: Alignment and Uniformity)
> **Original Paper Text:**
> *"As motion clips can be arbitrarily long, we split the data into overlapping sub-motions of 2 seconds. This results in motions that are long enough to present distinct and coherent characteristics. Additionally, sub-motions with close temporal proximity are likely to have similar characteristics. We leverage this understanding to improve the latent space structure by applying an alignment and uniformity loss on the predicted embeddings [Wang and Isola 2020].
> $$\mathcal{L}_{\text{align}} = \mathbb{E}_{(M, M') \sim \mathcal{M}_{\text{overlapping}}} \left[ \| E(M) - E(M') \|^2 \right], \quad (5)$$
> $$\mathcal{L}_{\text{uniform}} = \log \mathbb{E}_{(M, M') \stackrel{\text{i.i.d}}{\sim} \mathcal{M}} \left[ \exp\left( -2 \| E(M) - E(M') \|^2 \right) \right], \quad (6)$$
> where overlapping corresponds to two-second sub-motions $(M, M')$ from the same original motion sequence with non-zero overlap, and i.i.d to sub-motions randomly sampled from the data."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Why 2-Second Windows?**
  At 30 Hz (or 50 Hz), 2 seconds is 60–100 frames. This spans at least one or two complete gait cycles (heel-strike $\to$ stance $\to$ toe-off $\to$ swing), capturing distinct cadence, stride, and posture.
* **Wang & Isola (2020) Contrastive Geometry:**
  - **Alignment Loss $\mathcal{L}_{\text{align}}$ (Equation 5):**
    Forces sub-motions that overlap in time (e.g. seconds [0, 2] and [0.5, 2.5] of the same sprint) to map to nearly identical points on the hypersphere. This enforces temporal smoothness and invariance to phase offset.
  - **Uniformity Loss $\mathcal{L}_{\text{uniform}}$ (Equation 6):**
    Derived from the Gaussian potential kernel on hyperspheres. It pushes randomly sampled pairs $(M, M')$ apart, preventing latent collapse where all motions map to a single cluster. It maximizes information entropy on $\mathbb{S}^{d-1}$.

#### 2. Unitree G1 Humanoid Control Relevance
* Alignment ensures that whether the G1 starts tracking a walking clip at left-heel strike or right-heel strike, the latent code $z$ remains identical.
* Uniformity ensures the full capacity of the 64-dimensional latent space is utilized, maximizing the richness of the G1's skill vocabulary.

---

### Paragraph 5.9 (Subsubsection 5.1.2: Skill Transitions)
> **Original Paper Text:**
> *"When performing new tasks, agents often need to sequentially transition between behaviors. Even in simple tasks, like reaching a location, the character needs to utilize a combination of skills like walking, turning, and standing. To enable smooth and robust transitions between motions, we explicitly train the model to transition between different motions by changing the conditional motion $M$ at random timesteps. This teaches the policy to successfully transition between disparate motions."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* In vanilla imitation learning, episodes are conditioned on a single static skill from $t = 0$ to $t = T$. The policy only learns the steady-state limit cycle of that skill.
* **CALM's Dynamic Switching:**
  During an episode rollout of $K$ steps, at random timesteps $t_{\text{switch}} \sim \text{Geometric}(p)$, a new motion $M_{\text{new}} \sim \mathcal{M}$ is sampled, and the latent input instantly switches:
  $$z_{t < t_{\text{switch}}} = E(M_{\text{old}}) \implies z_{t \ge t_{\text{switch}}} = E(M_{\text{new}})$$
* The policy is suddenly confronted with an initial state $s_{t_{\text{switch}}}$ that belongs to $M_{\text{old}}$ (e.g. mid-sprint) while the discriminator begins rewarding $M_{\text{new}}$ (e.g. crouching).
* To maximize reward, the policy is forced to learn **dynamically feasible transition trajectories** that brake, redirect momentum, and stabilize into the new posture without falling.

#### 2. Unitree G1 Humanoid Control Relevance
* Bipedal balance during sudden transitions is the most dangerous failure mode on the Unitree G1. If high-level navigation suddenly switches from full sprint to abrupt stop, high forward momentum will tip the robot onto its face unless it performs rapid braking steps.
* CALM's random switching pre-trains this reactive recovery behavior directly in simulation.

---

### Paragraph 5.10 (Subsubsection 5.1.3: Discriminative Loss & Gradient Penalty)
> **Original Paper Text:**
> *"In adversarial imitation learning the discriminative reward provides the learning signal to the agent. To improve training dynamics, [Juravsky et al. 2022; Peng et al. 2022, 2021] propose to use a gradient penalty regularizer and negative sampling. This regularization helps mitigate discriminator overfitting and produces a smoother optimization landscape for the policy. The result is improved training stability and overall quality of the generated motion. In addition, as our goal is to learn a motion encoding optimal for the control task, we prevent gradients from flowing from the discriminator’s objective into the encoder. This results in the following objective:
> $$\mathcal{L}_D = -\mathbb{E}_{M} \left[ \mathbb{E}_{d^M} [\log \mathcal{D}(\hat{s}, \hat{s}' | z)] + \mathbb{E}_{d^\pi} [\log(1 - \mathcal{D}(s, s' | z))] + w_{\text{gp}} \mathbb{E}_{d^M} [\| \nabla_\phi \mathcal{D}(\phi | z) \|^2] + \mathbb{E}_{d^{\bar{M}}} [\log(1 - \mathcal{D}(\bar{s}, \bar{s}' | z))] \Big| z = \text{sg}(E(M)) \right], \quad (7)$$
> Here, $w_{\text{gp}}$ is the gradient penalty coefficient."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Gradient Penalty (WGAN-GP / AMP style):**
  $$w_{\text{gp}} \mathbb{E}_{d^M} [\| \nabla_\phi \mathcal{D}(\phi | z) \|^2]$$
  Penalizes the discriminator for having steep gradients around the manifold of true demonstrations. This prevents the discriminator from becoming a step function (which would provide zero informative gradient $\nabla_s \mathcal{D} \approx 0$ to the policy). In CALM, $w_{\text{gp}} = 10$.
* **Negative Sampling ($\bar{M} \neq M$):**
  $$\mathbb{E}_{d^{\bar{M}}} [\log(1 - \mathcal{D}(\bar{s}, \bar{s}' | z))]$$
  Samples real transitions $(\bar{s}, \bar{s}')$ from a *different* motion $\bar{M}$, but feeds them to the discriminator paired with latent $z = E(M)$. The discriminator must classify them as FALSE. This prevents the discriminator from collapsing into an unconditioned AMP discriminator that accepts any real human motion!
* **Stop-Gradient Operation $\text{sg}(E(M))$:**
  **Crucial Architectural Decision!**
  Notice $z = \text{sg}(E(M))$ in Equation (7). Gradients from $\mathcal{L}_D$ are **blocked** from flowing back into the encoder.
  * *Why?* If $\mathcal{L}_D$ were allowed to backprop into $E(M)$, the discriminator and encoder could cheat by collapsing $E(M)$ into degenerate representations that make classification trivially easy. $E(M)$ must only be shaped by policy control gradients and alignment/uniformity losses.

#### 2. Unitree G1 Humanoid Control Relevance
* Training adversarial RL in Isaac Gym with 4096 robots is prone to exploding discriminators and training instability.
* Gradient penalties and stop-gradient on $E(M)$ are mandatory to achieve stable convergence over 5 billion simulation steps for the G1.

---

## 6. High-Level Control

### Paragraph 6.1 (Section 6 Intro: Grounding High-Level Policy in Motor Priors)
> **Original Paper Text:**
> *"Once the low-level controller has been trained, it is used as a motion generator, grounding the motions of the character to those seen in the data. In this section, we present how to train a high-level policy to control the direction in which motions are performed, which can then be leveraged to solve complex tasks in varying forms without specifically training on them."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Freezing the Low-Level Controller:**
  In Phase 2, the low-level policy $\pi(a|s, z)$ and the encoder $E(M)$ are frozen.
* The high-level policy $\pi^H(z_t | s_t, g)$ treats the simulated robot + frozen low-level policy as an augmented MDP:
  - High-level action space: $\mathcal{Z} = \mathbb{S}^{63}$.
  - Low-level policy handles all balance, joint coordination, and ground contact dynamics.
  - High-level policy only needs to learn navigation vectors.

#### 2. Unitree G1 Humanoid Control Relevance
* On the G1, this provides complete safety isolation. Even if the high-level steering policy makes erratic decisions, the low-level policy guarantees the robot remains dynamically balanced and does not execute mechanically impossible joint motions.

---

### Paragraph 6.2 (Subsection 6.1: Precision Training & Directional Heading Reward)
> **Original Paper Text:**
> *"We achieve this by combining a task reward with a latent similarity loss. Specifically, given a motion encoding $\hat{z} = E(M)$, we train the high-level policy with the following reward
> $$r_t^{\text{locomotion}} = \exp\left( -0.25 \left\| d_t^* - \frac{\dot{x}_t^{\text{root}}}{\| \dot{x}_t^{\text{root}} \|} \right\|^2 \right) + \exp\left( -4 \| z_t - \hat{z} \|^2 \right), \quad (8)$$
> where $\dot{x}_t^{\text{root}}$ is the character’s velocity."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Anatomy of Equation (8):**
  - **Heading Alignment Term:**
    $$r_{\text{heading}} = \exp\left( -0.25 \left\| d_t^* - \frac{\dot{x}_t^{\text{root}}}{\| \dot{x}_t^{\text{root}} \|} \right\|^2 \right)$$
    Measures the cosine distance between desired heading unit vector $d_t^* \in \mathbb{R}^2$ and root velocity unit vector. Reaches $1.0$ when the character moves exactly in the commanded direction.
  - **Style Preservation Penalty:**
    $$r_{\text{style}} = \exp\left( -4 \| z_t - \hat{z} \|^2 \right)$$
    Penalizes the high-level policy if it deviates too far from the reference motion encoding $\hat{z}$.
* **The Balancing Act:**
  If the style penalty is too strong, the agent cannot turn (because the original MoCap clip was walking straight).
  If the style penalty is too weak, the agent abandons the style (e.g. switches from crouch-walking to standard running).
  The exponential weighting (coefficient $-4$) allows small local modulations of $z_t$ around $\hat{z}$ to steer the gait while strictly preserving the visual style.

#### 2. Unitree G1 Humanoid Control Relevance
* For G1, this enables **style-conditioned omnidirectional locomotion**:
  You pass $\hat{z}_{\text{patrol}}$ and stream joystick heading $d_t^*$. The G1 steers forward, backward, or diagonally while strictly maintaining the arm posture and foot-lifting characteristics of the patrol gait.

---

### Paragraph 6.3 (Subsection 6.2: Exemplar Guidance & FSM Task Design)
> **Original Paper Text:**
> *"Given a pre-trained encoder and low-level policy (Section 5) and a pre-trained high-level policy (Section 6.1), the task designer describes how a task should be solved, in a natural and intuitive way, overcoming the fragility of reward design. Here, the task designer provides demonstrations for the various motions the agent should perform, enclosed within a finite-state machine (FSM) that determines when to transition between behaviors. For instance, the task of striking an object can be broken down into three phases 'run towards the object', once within 0.5 meters then 'perform an attack', and then 'stand idle' until the next command."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* Traditional RL for complex missions requires meticulously tuning sparse and dense rewards ($r = w_1 r_{\text{reach}} + w_2 r_{\text{strike}} + w_3 r_{\text{balance}}$), which frequently leads to **reward hacking** (e.g. the robot vibrates near the target to accumulate proximity rewards without striking).
* CALM decomposes missions into modular discrete states. State transitions are governed by deterministic geometric conditions (e.g. Euclidean distance $< 0.5\text{ m}$).

#### 2. Unitree G1 Humanoid Control Relevance
* In a warehouse or factory setting with Unitree G1, mission rules are clear:
  1. *Approach Pallet:* High-level policy steers G1 towards pallet coordinates.
  2. *Trigger:* Proximity sensor detects pallet at $0.4\text{ m}$.
  3. *Grasp Action:* Low-level policy fed latent $\hat{z}_{\text{lift}}$ executes two-arm box lift.
  4. *Transport:* High-level policy steers G1 to conveyor belt.
* All components are robust, deterministic, and modular.

---

### Paragraph 6.4 (Precision vs Versatility in FSM Control)
> **Original Paper Text:**
> *"Achieving such a level of control requires a combination of versatility, provided by the low-level policy, and precision, provided by a pre-trained high-level policy. During phases requiring precision, such as moving in a specified direction, the FSM provides the high-level policy with a requested motion embedding $\hat{z}$ and a direction in which this motion should be performed. When transitioning to isolated motions, such as a specific sword swipe, the FSM provides the motion encoding directly to the low-level policy.
> This enables re-usability without re-training and resembles how a user would interact with the character given a game controller. A single combination of (a) low-level policy, (b) encoder, and (c) high-level policy, are used to solve unseen tasks in varying forms."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* Two pathways of latent routing:
  1. *Closed-Loop Precision Route:* $\text{FSM} \to \pi^H(s_t, d^*, \hat{z}) \to z_t \to \pi(a_t | s_t, z_t) \to \text{Robot}$. (Used for continuous steering and navigation).
  2. *Open-Loop Direct Route:* $\text{FSM} \to \hat{z}_{\text{motion}} \to \pi(a_t | s_t, \hat{z}) \to \text{Robot}$. (Used for self-contained, ballistic motor actions like kicking, striking, jumping, or celebrating).

#### 2. Unitree G1 Humanoid Control Relevance
* This dual-route architecture matches how roboticists control humanoids in real deployments:
  - Continuous locomotion is regulated by closed-loop heading and velocity feedback.
  - Reflexive or atomic manipulation primitives (e.g., press button, punch, stomp) are triggered directly as open-loop latent skill bursts.

---

# PART IV: EXPERIMENTS, RESULTS, LIMITATIONS & DISCUSSION

---

## 7. Experiments

### Paragraph 7.1 (Motion Dataset Details)
> **Original Paper Text:**
> *"The data: To acquire diverse motion control capabilities, the low-level policy is trained using 160 motion clips totaling over 30 minutes in duration [Reallusion, 2022]. Each motion clip is broken down into 2-second continuously overlapping sub-sequences, oblivious to transition boundaries. Hence, if a motion clip contains multiple skills, the 2-second clips may contain motions from several skills. This includes basic movements such as various forms of walking, as well as more complex motions such as sword-strike combinations. This is explained in further detail in the supplementary material."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Scale of Dataset:** 160 clips, >30 minutes of 30 Hz MoCap $\approx 54,000$ unique frames.
* **Sliding Window Slicing:**
  Clips are sliced into 2-second overlapping windows with a stride (e.g. 0.1 s). This generates hundreds of thousands of sub-sequences.
* **Boundary Invariance:**
  The authors do not manually cut clips at exact skill boundaries (e.g. "where the walk ends and the jump starts"). Sub-windows often straddle multiple phases. This teaches the encoder and policy to handle hybrid and composite motions naturally.

#### 2. Unitree G1 Humanoid Control Relevance
* For G1 training, this means you can pool together large open-source datasets (e.g. AMASS, CMU, 100Style) without tedious manual segmentation. Slicing with a 2-second sliding window creates a rich continuous training distribution for the G1.

---

### Paragraph 7.2 (Training Workflow & PPO Optimization)
> **Original Paper Text:**
> *"Training workflow: Throughout the pre-training process, the agent interacts with the environment for rollouts of $K$ steps. At random timesteps, a random motion is sampled from the reference dataset, it is then encoded and provided to the low-level policy. The rollout then consists of the observed states resulting from the low-level policy, conditioned on the latent $z$, interacting with the environment. The low-level policy is then trained with respect to the collected rollout using PPO [Schulman et al., 2017]."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Rollout Horizon $K$:** In Isaac Gym, parallel environments collect rollouts of $K = 32$ steps.
* **PPO Optimization:**
  Standard clipped surrogate objective:
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$
  where $\hat{A}_t$ is computed using Generalized Advantage Estimation (GAE) with $\lambda = 0.95$ and $\gamma = 0.99$.
* The adversarial reward $r_t$ from the conditional discriminator is fed directly as the environmental reward to PPO.

#### 2. Unitree G1 Humanoid Control Relevance
* PPO is the gold standard for humanoid locomotion RL due to its monotonic improvement guarantees and resilience to high-variance contact transitions on bipedal feet.

---

### Paragraph 7.3 (Hardware Infrastructure & Simulation Scale)
> **Original Paper Text:**
> *"We parallelize training over 4096 Isaac Gym [Makoviychuk et al., 2021] environments on a single A100 GPU, for a total of 5 billion steps. The low-level (high-level) policy takes decisions at a rate of 30 (6) Hz. The latent space $\mathcal{Z}$ is defined as a 64D hypersphere."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Massive Parallelism via Isaac Gym:**
  4096 simulated humanoids run in parallel on a single NVIDIA A100 GPU with end-to-end GPU physics (PhysX GPU pipeline).
  Tensor memory avoids CPU-GPU copying, enabling throughput of $>100,000$ simulation steps per second.
  5 billion steps require ~2–3 days of wall-clock training time.
* **Control Frequencies:**
  - Low-level policy: 30 Hz ($dt = 0.0333\text{ s}$). Simulator substeps run at 120 Hz (4 substeps per policy step).
  - High-level policy: 6 Hz (every 5 low-level steps).
* **Latent Space Dimension:** $d = 64$. Large enough to encode 160 distinct motions without dimensional bottleneck, but compact enough for stable hypersphere geometry.

#### 2. Unitree G1 Humanoid Control Relevance
* For the Unitree G1:
  - We recommend running low-level policy at 50 Hz ($dt = 0.02\text{ s}$), matching Unitree's standard control loop frequency.
  - Simulator physics should run at 200–500 Hz (4 to 10 substeps) to accurately capture foot contact stiffness and actuator rotor inertia.
  - 64D latent hypersphere is perfectly sized for G1's 23-DOF kinematic configuration.

---

### Paragraph 7.4 (Neural Network Architectures)
> **Original Paper Text:**
> *"Model architecture: The encoder is a standard MLP, mapping $E(M) \mapsto z$, the policy and conditional discriminator each contain an additional input head $H(z)$ for latent parsing, followed by an MLP $\pi(s, H(z)) \mapsto a$, where $a \in \mathbb{R}^{31}$."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Encoder MLP:**
  Input: Flattened 3D joint trajectories of 2-second sub-motion ($60 \text{ frames} \times 15 \text{ joints} \times 3 = 2700\text{ dimensions}$).
  Architecture: Linear(2700 $\to$ 512) $\to$ ReLU $\to$ Linear(512 $\to$ 256) $\to$ ReLU $\to$ Linear(256 $\to$ 64) $\to$ $L_2$ Normalization.
* **Latent Parsing Head $H(z)$:**
  A separate 2-layer MLP (256, 128 units) transforms raw latent $z$ into feature vector $h_z$.
* **Policy Trunk:**
  Concatenates physical state $s$ with $h_z$, passing through MLP (1024, 512 units) $\to$ Mean action $a \in \mathbb{R}^{31}$ + diagonal standard deviation.
* **Discriminator Trunk:**
  Concatenates $(s, s')$ with $h_z^D$, passing through MLP (1024, 512 units) $\to$ 1 scalar logit.

#### 2. Unitree G1 Humanoid Control Relevance
* For G1 (23 actuated DOFs):
  Action output $a \in \mathbb{R}^{23}$.
  State vector $s \in \mathbb{R}^{75}$ (pelvis height, orientation, angular velocity, linear velocity, 23 joint positions, 23 joint velocities).
  The $H(z)$ parsing head is crucial: it prevents the high-dimensional state vector from drowning out the latent vector during early training.

---

## 8. Results

### Paragraph 8.1 (Section 8 Intro: Evaluation Axes)
> **Original Paper Text:**
> *"We tested the effectiveness of our method by using CALM to learn skill embeddings that allow a simulated humanoid to perform various motion control tasks. We demonstrate that CALM learns a semantically meaningful latent representation of diverse human motion and a directable policy. We then trained a high-level policy to control the direction in which these motions are performed. Finally, we show how these low and high-level policies can be re-used for solving unseen tasks without further training. See the motions produced by CALM in the provided supplementary video."*

---

### Paragraph 8.2 (Subsection 8.1: Quantitative Comparison with ASE - Table 1)
> **Original Paper Text:**
> *"We begin by analyzing three aspects of CALM: (1) the encoder quality, (2) diversity of the low-level controller, and (3) controllability of the combined system. Results are reported in Table 1. We focus our comparison on ASE [Peng et al. 2022], a latent generative model which learns to map arbitrary latent variables to motions.
> Experiment 1: Encoder quality. Using Fisher’s class separability metric [Bishop et al. 1995] over the representation learned by the encoder, we measure the separability between the motion classes within the latent space, where a motion class is defined as sub-motions within a single motion file. As shown in Table 1, CALM learns to encode motions into representations with much better separation."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Table 1 Quantitative Results:**
  | Method | Encoder Quality (Fisher $\downarrow$) | Diversity (Inception $\uparrow$) | Controllability (Accuracy $\uparrow$) |
  | :--- | :---: | :---: | :---: |
  | **CALM (Ours)** | **0.23** | **19.8 ± 0.1** | **78%** |
  | **ASE** | 0.68 | 18.6 ± 0.4 | 35% |
* **Metric 1: Fisher's Concentration / Class Separability (Encoder Quality):**
  Measures the ratio of within-class scatter $S_W$ to between-class scatter $S_B$:
  $$F = \frac{\text{Tr}(S_W)}{\text{Tr}(S_B)}$$
  Lower values mean clips of the same skill cluster tightly together, while different skills are well-separated. CALM achieves **0.23 vs 0.68 for ASE** (nearly a $3\times$ improvement).
* **Metric 2: Inception Score for Motion (Diversity):**
  Measures how many distinct, identifiable motion classes the policy can generate across random latents:
  $$\text{IS} = \exp\left( \mathbb{E}_{z} [D_{\text{KL}}(p(y | x) \parallel p(y))] \right)$$
  CALM matches and slightly exceeds ASE (19.8 vs 18.6).
* **Metric 3: Generation Accuracy (Controllability):**
  When given latent $z = E(M_{\text{target}})$, does the generated simulated motion actually match $M_{\text{target}}$ when classified by an independent motion classifier?
  **CALM achieves 78% vs ASE's 35% — more than double the controllability!**

#### 2. Unitree G1 Humanoid Control Relevance
* A 35% controllability score (ASE) is completely unacceptable for robotics: commanding G1 to walk would result in the wrong motion 65% of the time!
* CALM's 78% zero-shot controllability guarantees that the commanded latent reliably maps to the expected physical gait on G1.

---

### Paragraph 8.3 (Subsubsection 8.1.1: Qualitative Analysis & Latent Interpolation)
> **Original Paper Text:**
> *"Qualitative analysis. In Figure 3, we show motions generated by CALM. Throughout a single episode, the conditional motions were changed, resulting in human-like transitions between the requested motions. Additionally, to illustrate the semantic structure of the latent space, we encode two semantically connected motions 'sprint' and 'crouching idle' and interpolate between their encodings over time. As shown in Figure 3e, CALM smoothly transitions between the two motions, decreasing both speed and height while continuously performing a form of walking motion."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Continuous Semantic Interpolation:**
  Let $z_{\text{sprint}} = E(M_{\text{sprint}})$ and $z_{\text{crouch}} = E(M_{\text{crouch}})$.
  Interpolating along the spherical arc $z(t) = \text{slerp}(z_{\text{sprint}}, z_{\text{crouch}}, t)$ does *not* produce erratic falling or joint freezing.
  Instead, the simulated character continuously modulates its height and forward velocity, seamlessly transitioning through a jog $\to$ walk $\to$ crouch-walk $\to$ crouch-idle.
* This proves that the latent space has a **smooth, convex-like dynamical manifold**.

#### 2. Unitree G1 Humanoid Control Relevance
* This continuous interpolation allows G1 to perform **continuous gait morphing**:
  Instead of hard-switching gaits, an operator can slide a joystick throttle from 100% to 0%, causing G1 to smoothly drop from a high-speed run into a low crouch without breaking bipedal balance or experiencing impact shock on its leg gearboxes.

---

### Paragraph 8.4 (Subsection 8.2: Solving Downstream Tasks)
> **Original Paper Text:**
> *"Using the encoder and low-level policy from Section 8.1, we show how they can be used to compose motions for solving unseen tasks."*

---

### Paragraph 8.5 (Subsubsection 8.2.1: Directional Motion Control - Table 2)
> **Original Paper Text:**
> *"Directional motion control. First, we show that provided a desired motion style, the agent can navigate in a requested direction. In this experiment, a target direction $d_t^*$ is provided to the agent and rotated every 5 seconds. We evaluate performance on three forms of locomotion: run, walk, and crouch-walk. For each motion, we measure the style score, which measures the similarity between the reference motion encoding and the predicted latents $z_t$, and the heading score, which measures the directional alignment between the requested and actual movement direction.
> As shown in Table 2, the high-level policy is capable of generating motion in the specified style while ensuring it moves in the requested direction."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Table 2 Quantitative Results:**
  | Motion Style | Style Score $\exp(-4\|z - \hat{z}\|^2)$ | Heading Score $\cos(\theta)$ |
  | :--- | :---: | :---: |
  | **Run** | **1.00** | **0.92** |
  | **Walk** | **0.81** | **0.92** |
  | **Crouch-Walk** | **0.94** | **0.91** |
* **Performance Analysis:**
  Across all three distinct locomotion gaits, the heading score remains above 0.91 (corresponding to an angular error of less than $24^\circ$, with immediate steering convergence within 0.5 seconds of command change).
  Simultaneously, the style score remains extremely high ($0.81 - 1.00$), proving that directional steering does not degrade the visual fidelity or dynamical profile of the gait.

#### 2. Unitree G1 Humanoid Control Relevance
* This directly validates that Unitree G1 can be steered in 2D space (omnidirectional velocity commands $v_x, v_y, \omega_z$) while maintaining arbitrary designated motion styles (e.g., carrying posture, crouching for low clearance, brisk walk).

---

### Paragraph 8.6 (Subsubsection 8.2.2: Solving Tasks Without Further Training - Location & Strike)
> **Original Paper Text:**
> *"Solving tasks without further training. In our final experiment, we combine the directable low-level controller together with the high-level locomotion policy to provide zero-shot solutions to unseen tasks. We consider two tasks, location and strike. For the location task, the agent should reach and remain within the goal position – illustrated as a circle around the flagpost. Strike, a more complex task, requires the agent to reach the target and strike it down. In both cases, the character is controlled by conditioning a sequence of reference motions. To do so, the direction vector is provided to the target location, represented in the character’s local coordinate frame.
> As seen in Table 2, the agent achieves near-perfect ending scores (0.96 – 1.00) across diverse combinations of locomotion styles (run, walk, crouch-walk) and finishing skills (stand idle, celebrate, crouch idle, kick, shield charge, sword swipe)."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Zero-Shot Composite Tasks:**
  The system is never trained on the composite objective "run to flag and celebrate".
  - **Location Task:**
    $d^* = \text{normalize}(x_{\text{target}} - x_{\text{root}})$.
    Phase 1: High-level policy steers toward target using $\hat{z}_{\text{run}}$.
    Phase 2: When $\|x_{\text{root}} - x_{\text{target}}\| < R$, switch directly to low-level policy fed $\hat{z}_{\text{celebrate}}$.
  - **Strike Task:**
    Phase 1: High-level policy steers toward dummy using $\hat{z}_{\text{crouch-walk}}$.
    Phase 2: When within 0.5m, trigger $\hat{z}_{\text{kick}}$. The foot collision knocks the target down.
* Ending scores of 0.96 – 1.00 prove that the transitions between high-level steering and atomic skill execution are physically seamless and robust.

#### 2. Unitree G1 Humanoid Control Relevance
* For G1, this proves that multi-step real-world robotics missions (e.g., "walk to button $\to$ push button $\to$ stand idle") do not require custom end-to-end RL training.
* You simply compose the pre-trained CALM locomotion policy and atomic manipulation primitives within an FSM.

---

## 9. Limitations

### Paragraph 9.1 (Limitations Intro)
> **Original Paper Text:**
> *"The results in Tables 1 and 2 show how CALM learns a diverse repertoire of motions without sacrificing controllability. This can then be leveraged to learn style-conditioned locomotion and finally to compose motions for solving multi-step, unseen, tasks. In this section, we highlight open challenges and questions that arise from this work."*

---

### Paragraph 9.2 (Pre-Training: Mode Collapse & Idle Micromotions)
> **Original Paper Text:**
> *"Pre-training – mode collapse: We have shown that our algorithm CALM improves the controllability of generated motions compared to the existing approach, ASE, with a significant boost in performance from 35% up to 78%. However, mode collapse remains an open challenge. For instance, we have observed that conditioning the low-level controller on idle-motions can lead to unrealistic micromotions that slowly move the character. Although our approach addresses the problem of mode collapse to a large extent, there remains room for further improvements."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **The Idle Drift Phenomenon:**
  When conditioned on an "idle standing" motion, the simulated character sometimes exhibits tiny, creeping foot shuffles that cause slow translational drift.
  *Why?* The adversarial discriminator penalizes deviations from the idle distribution, but because the actor is a continuous neural network outputting PD targets, small residual biases in joint predictions produce microscopic net forward forces against ground friction.

#### 2. Unitree G1 Humanoid Control Relevance
* On the physical Unitree G1, idle drifting is dangerous: the robot might slowly creep towards a table edge or obstacle while ostensibly in "standby".
* *Robotics Fix:* Implement an explicit zero-velocity foot clamp or switch the leg motors to high-damping stationary PD mode ($K_d \dot{q}$) when the commanded skill is idle standing.

---

### Paragraph 9.3 (Pre-Training: Unseen Out-of-Distribution Motions)
> **Original Paper Text:**
> *"Pre-training – unseen motions: Our work focuses on learning a latent generative motion controller for in-distribution motions. However, when conditioning the character on encodings from unseen motions, we cannot guarantee the quality of the generated motions. While we have observed that some unseen motions map to semantically similar motions from the data, such as tip-toe mapping to bounce-walk, we anticipate that the model may fail as the motions become increasingly out of distribution."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* If you feed an encoder $E(M_{\text{breakdance}})$ when the training dataset only contained walking and sword fighting, the encoder will project it somewhere on $\mathbb{S}^{63}$, but the policy has never explored that region under that specific transition objective. The character may stumble or collapse.

#### 2. Unitree G1 Humanoid Control Relevance
* On G1, you must restrict the motion library fed to $E(M)$ to kinematically verified skills. Before deploying a new MoCap clip to the physical robot, verify its latent trajectory $E(M)$ in simulation.

---

### Paragraph 9.4 (Precision Training: Beyond Locomotion)
> **Original Paper Text:**
> *"Precision-training – beyond locomotion: In Table 2 our approach is shown to leverage the learned latent space and achieve style-conditioned locomotion. However, controlling intricate movements such as the path of a sword or shield in an attack may require additional innovations in the pre-training phase, such as learning motions with a larger distributional discrepancy to the data."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* Precision training in Section 6.1 focused on 2D base heading $d^*$.
* Extending precision training to 6-DOF end-effector tracking (e.g. putting a sword tip through a 5 cm ring) requires high-dimensional spatial conditioning that conflicts with maintaining the broad stylistic manifold.

#### 2. Unitree G1 Humanoid Control Relevance
* For G1 manipulation (e.g. grasping a cup or opening a door handle), pure latent style steering is insufficient.
* *Solution:* Decouple the G1: use CALM for the lower body (bipedal balance, locomotion base) and use an operational space controller (OSC) or specialized manipulation policy for the 7-DOF arms and dexterous hands.

---

### Paragraph 9.5 (FSM Robustness Envelope & Complex Terrains)
> **Original Paper Text:**
> *"FSM – robustness: Our approach has demonstrated the ability to solve tasks using classic tools from the gaming and animation industry, such as FSM and behavior trees, presenting a game-controller-like interface, as shown in Figure 5. However, we anticipate that the policy’s robustness envelope may limit its ability to solve tasks with vastly different dynamics from those seen during training, such as climbing stairs or walking on uneven terrain. Therefore, further innovations and training may be required to solve such tasks."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* The low-level policy in CALM was trained on flat planar ground.
* When faced with 20 cm stairs or rough rubble, the foot contact heights violate the flat-ground distribution $d^M(\hat{s}, \hat{s}')$. The policy lacks elevation perception.

#### 2. Unitree G1 Humanoid Control Relevance
* To deploy CALM on Unitree G1 outdoors or on stairs:
  - Add heightmap / depth sensor observations to the state $s_t$.
  - Train in Isaac Gym with rough terrain generators (stepping stones, stairs, slopes).
  - Include terrain-aware MoCap or parkour demonstrations in $\mathcal{M}$.

---

### Paragraph 9.6 (Rendering Artifacts vs Physical Simulation)
> **Original Paper Text:**
> *"Rendering – artifacts: To illustrate how our work can be integrated into the gaming industry, we visualize the motions using high-resolution characters, rendered within Omniverse (OV). Although physically accurate motion is recorded in IsaacGym (IG) where physical constraints are maintained without visual artifacts, the visualization character in OV is not subject to these constraints during rendering. Consequently, due to the difference in character geometry, the visualization character may exhibit penetrations and other visual artifacts that do not occur in IG. It is worth noting that the issue of rendering artifacts is not an inherent problem with our proposed algorithm CALM, but due to the differences in character geometry. One way to minimize these artifacts is by ensuring that the simulated character’s geometry is closer to that of the visualization. Another solution is robustifying the training process to handle varying character morphologies to directly control the visualization character while enforcing physical constraints."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* In graphics, a simplified collision capsule model is used in Isaac Gym, while a high-polygon skinned mesh is rendered in Omniverse. Discrepancies between capsule radii and mesh skinning create apparent self-penetrations.

#### 2. Unitree G1 Humanoid Control Relevance
* In robotics, this is the **Sim-to-Real URDF Discrepancy**:
  The CAD model in Isaac Gym has ideal rigid links and collision meshes. The real G1 has wiring harnesses, battery bulge, and slight link compliance. Accurate collision geometry and URDF inertial calibration are necessary to prevent unmodeled self-collisions.

---

## 10. Discussion and Future Work

### Paragraph 10.1 (Summary of CALM Capabilities)
> **Original Paper Text:**
> *"In this work, we presented CALM, a framework for learning reusable and directable motor skills for physics-based character animation. Our model enables the character’s behaviors to be directed using motion clips. Given an unlabeled motion dataset, CALM learns both an encoder and a low-level controller. The encoder maps motions onto a semantically meaningful low-dimensional representation and a low-level controller takes the role of a decoder and produces motions with similar characteristics to those encoded within the learned representation. These reference motions can be used both for controlling low-level skills and to guide higher-level controllers and specify which motions the character should use when solving complex tasks. The ability to control the generated motion of the character enables zero-shot solutions to complex multi-step tasks, a step towards real integration of interactive virtual characters."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* Comprehensive recap: Unlabeled MoCap $\to$ Joint Encoder + Conditional Discriminator $\to$ Structured Latent Manifold $\to$ High-Level Precision Steering $\to$ Zero-Shot FSM Composition.

#### 2. Unitree G1 Humanoid Control Relevance
* Establishes the foundation for Hierarchical Behavioral Control (HBC) on humanoids.

---

### Paragraph 10.2 (Disentangling Direction from Content Motion)
> **Original Paper Text:**
> *"Our motion-constrained training enabled guiding the solution towards utilizing pre-specified motions. However, successfully learning to produce the requested motion in the specified direction required delicate tuning of the reward parameters. We are interested in exploring ways for disentangling the representation of direction from the representation of the content motion. Such disentanglement will enable high-level policies to learn with simplified rewards, while ensuring that they produce motion with the desired characteristics."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* Currently, the high-level policy must trade off heading alignment with style penalty $\|z - \hat{z}\|^2$ (Equation 8).
* If the latent space $\mathcal{Z}$ could be factored into $\mathcal{Z}_{\text{style}} \times \mathcal{Z}_{\text{direction}}$ (e.g. via Lie group representations or orthogonal latent subspaces), heading could be modulated completely independently without affecting style.

#### 2. Unitree G1 Humanoid Control Relevance
* On the G1, this would allow clean separation: the teleoperation joystick directly commands $\mathcal{Z}_{\text{direction}}$, while an autonomous AI agent or user interface selects $\mathcal{Z}_{\text{style}}$ (e.g. "stealth mode", "carry mode").

---

### Paragraph 10.3 (Object Interaction & Environmental Affordances)
> **Original Paper Text:**
> *"Finally, some motions require coordinated interaction with the environment. For instance, acrobatic motions like handsprings and vaults can only be performed while interacting with a vaulting table/elevated box. We intend to investigate automation methods for understanding the motion-object pairs, and the integration of such objects throughout training for learning the respective motions."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* MoCap data of human vaulting or sitting assumes an external physical prop. Training a character without that prop in the simulator causes immediate physics failure. Conditioning must include object geometry and contact affordances.

#### 2. Unitree G1 Humanoid Control Relevance
* For G1 loco-manipulation (e.g. pushing a cart, opening a door, climbing onto a pallet), the low-level policy must be trained with parameterized object interactions in simulation.

---

# PART V: APPENDIX DEEP-DIVE & DETAILED FORMULATIONS

---

## Appendix A. State and Action Space

### A.1. Low-Level Policy Space Formulation
> **Original Paper Text:**
> *"In this work, we consider a 3D physically-simulated humanoid character wielding a sword and a shield, with 37 degrees of freedom. A similar character was used in Peng et al. [2022]. To encode the character’s state, we follow previous work and represent the state using the character’s root position along the height axis, root orientation, linear and angular velocities, as well as joint positions and velocities. As the sword and shield are rigidly attached to the character’s hands, they are not explicitly modeled in the state space."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **State Vector Formulation $s_t$:**
  $$s_t = \Big[ h_{\text{root}}, \, q_{\text{root}}, \, \dot{x}_{\text{root}}^{\text{local}}, \, \omega_{\text{root}}^{\text{local}}, \, \{ q_j \}_{j=1}^J, \, \{ \dot{q}_j \}_{j=1}^J \Big]$$
  - Root height $h_{\text{root}} \in \mathbb{R}^1$.
  - Root orientation $q_{\text{root}} \in \mathbb{R}^4$ (unit quaternion) or 6D continuous rotation representation.
  - Local root linear velocity $\dot{x}_{\text{root}}^{\text{local}} \in \mathbb{R}^3$ and angular velocity $\omega_{\text{root}}^{\text{local}} \in \mathbb{R}^3$.
  - Relative joint positions $q_j$ and angular velocities $\dot{q}_j$.
* **Action Vector Formulation $a_t$:**
  - $a_t \in \mathbb{R}^{31}$: Target joint angles $q_{\text{target}} = q_0 + a_t$ fed into proportional-derivative (PD) servos.
  - Joint torque: $\tau = K_p(q_{\text{target}} - q) - K_d \dot{q}$.
  - Gains: $K_p$ and $K_d$ tuned per joint based on limb mass and gear ratio.

#### 2. Unitree G1 Humanoid Control Relevance
* For the Unitree G1 (23-DOF base version):
  - 12 leg DOFs: 6 per leg (hip pitch, hip roll, hip yaw, knee pitch, ankle pitch, ankle roll).
  - 3 torso DOFs: waist yaw, waist roll, waist pitch.
  - 8 arm DOFs: 4 per arm (shoulder pitch, shoulder roll, shoulder yaw, elbow pitch).
* G1 State Vector $s_t^{\text{G1}} \in \mathbb{R}^{75}$:
  $$s_t^{\text{G1}} = [ z_{\text{pelvis}}, \, \text{rot6d}(\text{pelvis}), \, v_{\text{pelvis}}^{\text{local}}, \, \omega_{\text{pelvis}}^{\text{local}}, \, q_{1:23}, \, \dot{q}_{1:23} ]$$
* G1 Action Vector $a_t^{\text{G1}} \in \mathbb{R}^{23}$: position offsets $\Delta q$ added to nominal standing posture $q_0$.

---

### A.2. Encoder Input Representation
> **Original Paper Text:**
> *"The encoder takes as input a 2-second sub-motion $M$. A sub-motion consists of a sequence of poses, where each pose is represented by the 3D local coordinates of the character’s key joints, relative to the root position and heading."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Root-Relative Normalization:**
  Let $x_j^{\text{world}}(t)$ be the global 3D position of joint $j$ at time $t$.
  Transform into root frame:
  $$\hat{x}_j(t) = R_{\text{root}}(t)^{-1} \left( x_j^{\text{world}}(t) - x_{\text{root}}^{\text{world}}(t) \right)$$
  where heading rotation $R_{\text{root}}(t)$ only cancels yaw around the gravity vector $z$.
* This ensures the encoder input is **translation-invariant and yaw-invariant**. A forward sprint looks identical whether performed facing North, East, South, or West.

#### 2. Unitree G1 Humanoid Control Relevance
* For G1, retargeted MoCap trajectories are stripped of global coordinates $(X, Y)$ and global yaw $\psi$. The encoder only sees the relative geometry of the robot's limbs, ensuring universal reuse across any starting location in the factory.

---

### A.3. Discriminator Input Representation
> **Original Paper Text:**
> *"The conditional discriminator $\mathcal{D}(s, s' | z)$ takes as input two consecutive states $(s_t, s_{t+1})$ and the latent variable $z$. State transitions are transformed into the local coordinate frame of the character at step $t$."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* Input to discriminator:
  $$\phi(s_t, s_{t+1}) = \left[ \hat{p}_j(t), \, \hat{p}_j(t+1), \, \hat{v}_j(t), \, \hat{\omega}_j(t) \right]_{j \in \text{KeyJoints}}$$
  paired with $z \in \mathbb{S}^{63}$.
* Using two consecutive frames $(s_t, s_{t+1})$ allows the discriminator to observe not just instantaneous posture, but **joint velocities, accelerations, and contact transitions**.

#### 2. Unitree G1 Humanoid Control Relevance
* Crucial for G1 feet: observing $(s_t, s_{t+1})$ allows the discriminator to penalize foot sliding (where foot position shifts while contact force is non-zero). The policy is forced to execute clean stance-to-swing foot liftoffs.

---

### A.4. High-Level Policy Tasks: Block, Reach, Locomotion
> **Original Paper Text:**
> *"Block: $r_t^{\text{block}} = \mathbf{1}_{\text{projectile blocked by shield}}$.
> Reach: $r_t^{\text{reach}} = \exp\left( -4 \| x_t^{\text{sword}} - x^* \|^2 \right)$.
> Location: $r_t^{\text{location}} = \exp\left( -0.5 \| x^* - x_t^{\text{root}} \|^2 \right)$.
> Strike: $r_t^{\text{strike}} = 1 - u_{\text{up}} \cdot u_t^*$, with termination if character touches target with any part other than sword."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* In Appendix E.3, the authors show that if one *chooses* to train a high-level policy with RL instead of an FSM, the CALM latent space enables fast convergence across all these tasks.
* Because the low-level controller handles stability, the high-level policy learns to block flying projectiles in just a few million steps by modulating $z$ to reposition the shield.

#### 2. Unitree G1 Humanoid Control Relevance
* Demonstrates that CALM is not limited to bipedal walking: upper-body manipulation (reaching, blocking, striking) can be modulated through the same unified latent space $\mathcal{Z}$.

---

## Appendix B. Architecture & Network Dimensions

### Summary Table of Neural Network Architectures
| Module | Input Dimension | Hidden Layers | Output Dimension | Normalization / Activation |
| :--- | :--- | :--- | :--- | :--- |
| **Motion Encoder $E(M)$** | $60 \times 15 \times 3 = 2700$ | $[512, 256]$ | $64$ | ReLU, $L_2$ Hypersphere Normalization |
| **Policy Latent Head $H_\pi(z)$** | $64$ | $[256, 128]$ | $128$ | ReLU |
| **Policy Trunk $\pi(s, H_\pi(z))$** | $\dim(s) + 128 = 203$ | $[1024, 512]$ | $31$ (Mean) + $31$ (Diag Std) | ReLU, Tanh / Gaussian Action |
| **Discriminator Latent Head $H_D(z)$** | $64$ | $[256, 128]$ | $128$ | ReLU |
| **Discriminator Trunk $\mathcal{D}(\phi, H_D(z))$**| $\dim(\phi) + 128$ | $[1024, 512]$ | $1$ Logit | ReLU, Linear (WGAN / cross-entropy) |
| **Value Function $V(s, z)$** | $\dim(s) + 64$ | $[1024, 512]$ | $1$ Scalar Value | ReLU, Linear |
| **High-Level Policy $\pi^H(s, d^*, \hat{z})$** | $\dim(s) + 2 + 64$ | $[512, 256]$ | $64$ (Mean) $\to L_2$ Norm | ReLU, $L_2$ Projection on $\mathbb{S}^{63}$ |

---

## Appendix C. Hyperparameters & Training Setup

### Training Hyperparameters
* **Simulation Engine:** NVIDIA Isaac Gym (PhysX GPU).
* **Parallel Environments:** 4096.
* **Low-Level Simulation Frequency:** 120 Hz ($dt_{\text{sim}} = 1/120\text{ s}$).
* **Low-Level Policy Frequency:** 30 Hz ($dt_{\text{policy}} = 1/30\text{ s}$, 4 physics substeps per policy step).
* **High-Level Policy Frequency:** 6 Hz ($dt_{\text{high}} = 1/6\text{ s}$, 5 low-level steps per high-level step).
* **PPO Clip Parameter $\epsilon$:** 0.2.
* **Discount Factor $\gamma$:** 0.99.
* **GAE Parameter $\lambda$:** 0.95.
* **Optimizer:** Adam (Learning rate $= 5 \times 10^{-5}$ for Policy, $10^{-4}$ for Discriminator, $10^{-4}$ for Encoder).
* **Gradient Penalty Coefficient $w_{\text{gp}}$:** 10.0.
* **Alignment Loss Weight $w_{\text{align}}$:** 0.1.
* **Uniformity Loss Weight $w_{\text{uniform}}$:** 0.1.
* **Total Training Steps:** 5 Billion environment transitions (~48 hours on a single NVIDIA A100 GPU).

---

## Appendix D & E. Ablation Analysis & Comparisons

### Table 3: Pre-Training Ablation Study
> **Original Paper Text:**
> *"Concentration $\downarrow$ (Fisher), Generation $\uparrow$ (Diversity), Accuracy $\uparrow$ (Controllability):
> CALM (Full): Concentration = 0.23, Generation = 19.8 ± 0.11, Accuracy = 78%
> w/o negative samples: Concentration = 0.24, Generation = 15.7 ± 0.07, Accuracy = 62%
> w/o negative samples & w/o regularization: Concentration = 0.35, Generation = 12.8 ± 0.05, Accuracy = 61%"*

#### 1. In-Depth Technical Breakdown & Core Intuition
* **Impact of Negative Sampling:**
  Removing negative sampling in Equation (7) drops controllability from **78% down to 62%**.
  Without negative samples, the discriminator only learns that simulated transitions $(s, s')$ look realistic, but fails to strictly enforce that they correspond to the *specific* conditioned latent $z$.
* **Impact of Alignment and Uniformity Regularization (Equations 5 & 6):**
  Removing latent space regularization drops generation diversity from **19.8 down to 12.8** and concentration degrades from **0.23 to 0.35**.
  Without Wang & Isola regularization, the encoder clusters motions haphazardly, causing dead zones in the latent hypersphere.

#### 2. Unitree G1 Humanoid Control Relevance
* Confirms that every component of CALM's loss suite (conditional discriminator + negative samples + hypersphere alignment/uniformity) is essential. Skipping negative samples or regularization severely cripples gait controllability on the robot.

---

## Appendix F. Latent Space Analysis

### Semantic Clustering & Distance Matrix
> **Original Paper Text:**
> *"To gain further insight into the learned representations, we analyze the structure of the latent space. We split the motions into 5 categories: Walking, Sword attacks, Shield attacks, Turning, and Idle.
> For each motion in the reference dataset, we encode it using each method’s respective encoder. We then calculate the average pairwise distance between groups. As seen in Figure 9, CALM clusters motion groups closer in the latent space... while ASE motions corresponding to walking are dispersed over the latent space."*

#### 1. In-Depth Technical Breakdown & Core Intuition
* In CALM's pairwise distance matrix (Figure 9a), the diagonal blocks have deep dark values (low pairwise Euclidean distance).
  - All walking variations (slow walk, brisk walk, bounce walk) cluster in one compact neighborhood.
  - All sword swings cluster in another neighborhood.
  - All shield charges cluster in a third neighborhood.
* In ASE (Figure 9b), the diagonal structure is washed out. Walking motions are randomly scattered all across the sphere.

#### 2. Unitree G1 Humanoid Control Relevance
* Semantic clustering enables **predictable gait interpolation** on G1:
  To increase walking speed on G1, the high-level policy simply navigates along the local geodesic connecting `slow_walk` to `fast_walk` on $\mathbb{S}^{63}$, without risking erratic transitions into a jump or strike.

---

# PART VI: COMPLETE UNITREE G1 IMPLEMENTATION BLUEPRINT

This section synthesizes the entire CALM paper into a production-ready engineering specification for controlling the **Unitree G1 Humanoid Robot**.

```mermaid
graph TD
    subgraph Data Pipeline
        AMASS[AMASS / CMU MoCap Dataset] -->|SMPL Bodies| IK[Kinematic Retargeting via Pinocchio]
        IK -->|Joint Trajectories q_t| G1Data[G1 Reference MoCap Library \mathcal{M}]
        G1Data --> Slicer[2-Second Sliding Window Slicer]
        Slicer --> MClips[Training Sub-Motions M]
    end

    subgraph Simulation Training (Isaac Lab / Isaac Gym)
        MClips --> Encoder[Motion Encoder E(M)]
        Encoder -->|z \in \mathbb{S}^{63}| Actor[Low-Level Policy \pi(a|s, z)]
        Encoder -->|z \in \mathbb{S}^{63}| Disc[Conditional Discriminator \mathcal{D}(s, s'|z)]
        Actor -->|\Delta q (23 DOFs)| PDSim[PD Motor Controller: 500Hz]
        PDSim --> Physics[Rigid Body Dynamics: Isaac Gym]
        Physics -->|State s_t, s_t+1| Disc
        Physics -->|State s_t| Actor
        Disc -->|Adversarial Reward r_t| PPO[PPO Optimizer]
    end

    subgraph Sim-to-Real Transfer to Physical G1
        Actor --> Deploy[TensorRT / ONNX Export]
        Deploy --> Jetson[G1 Onboard Jetson Orin]
        StateEst[EKF State Estimator / IMU] --> Jetson
        Joystick[Teleop / ROS 2 Navigation] --> HighLevel[High-Level Precision Steering \pi^H]
        HighLevel -->|z_t| Jetson
        Jetson -->|Joint Targets q*| G1Hardware[Unitree G1 Actuators @ 500Hz]
    end
```

---

### Step 1: Kinematic Retargeting Pipeline (SMPL $\to$ Unitree G1)
1. **Source Data:** AMASS (SMPL body format, 52 joints).
2. **Target Kinematic Tree:** Unitree G1 URDF (23 actuated revolute joints, floating base pelvis).
3. **Optimization Problem:**
   Solve inverse kinematics (IK) frame-by-frame using Pinocchio or CasADi:
   $$\min_{q_t, \dot{q}_t} \sum_{j \in \text{EndEffectors}} w_j \| p_j^{\text{G1}}(q_t) - p_j^{\text{SMPL}}(t) \|^2 + w_{\text{reg}} \| q_t - q_{\text{default}} \|^2 + w_{\text{smooth}} \| \ddot{q}_t \|^2$$
   subject to:
   $$q_{\text{min}} \le q_t \le q_{\text{max}}, \quad |\dot{q}_t| \le \dot{q}_{\text{max}}$$
4. **Physical Feasibility Filtering:**
   Discard any retargeted frames that exceed G1's actuator limits (e.g. knee velocity $> 20\text{ rad/s}$ or ground penetration $> 2\text{ cm}$).

---

### Step 2: G1 State, Action, and Observation Spaces

#### Low-Level Policy Observation $s_t^{\text{G1}} \in \mathbb{R}^{78}$:
* **Base Height:** $z_{\text{pelvis}} \in \mathbb{R}^1$.
* **Base Orientation:** Projected gravity vector in body frame $R_{\text{body}}^T \mathbf{g} \in \mathbb{R}^3$ (avoids yaw drift).
* **Base Velocities:** Linear velocity $v_{\text{body}} \in \mathbb{R}^3$, Angular velocity $\omega_{\text{body}} \in \mathbb{R}^3$.
* **Joint Positions:** $q_j - q_j^{\text{default}} \in \mathbb{R}^{23}$.
* **Joint Velocities:** $\dot{q}_j \in \mathbb{R}^{23}$.
* **Previous Action:** $a_{t-1} \in \mathbb{R}^{23}$.
* **Foot Contact Flags:** Left foot, Right foot binary contacts $\in \{0, 1\}^2$.

#### Low-Level Action $a_t^{\text{G1}} \in \mathbb{R}^{23}$:
* Scaled joint position targets:
  $$q_t^{\text{target}} = q^{\text{default}} + 0.25 \cdot a_t^{\text{G1}}$$
* Low-level motor PD loop (500 Hz on hardware):
  $$\tau_j = K_{p, j} (q_{j, t}^{\text{target}} - q_{j, t}) - K_{d, j} \dot{q}_{j, t}$$
* **Default G1 PD Gains:**
  - Leg Hip/Knee: $K_p = 100.0 - 150.0\text{ N}\cdot\text{m/rad}$, $K_d = 2.0 - 4.0\text{ N}\cdot\text{m}\cdot\text{s/rad}$.
  - Ankle: $K_p = 40.0\text{ N}\cdot\text{m/rad}$, $K_d = 1.0\text{ N}\cdot\text{m}\cdot\text{s/rad}$.
  - Torso: $K_p = 100.0\text{ N}\cdot\text{m/rad}$, $K_d = 3.0\text{ N}\cdot\text{m}\cdot\text{s/rad}$.
  - Arms: $K_p = 40.0\text{ N}\cdot\text{m/rad}$, $K_d = 1.0\text{ N}\cdot\text{m}\cdot\text{s/rad}$.

---

### Step 3: Sim-to-Real Domain Randomization Suite
To guarantee zero-shot transfer from Isaac Gym to the physical Unitree G1, apply rigorous domain randomization during Phase 1 pre-training:
1. **Inertial Properties:**
   - Link mass: $\pm 10\%$.
   - CoM position: $\pm 1.5\text{ cm}$ along each axis.
2. **Ground Contact:**
   - Friction coefficient $\mu \in [0.4, 1.2]$.
   - Restitution $\in [0.0, 0.1]$.
3. **Actuator & Drive Dynamics:**
   - Motor torque limits: $\pm 10\%$.
   - PD gain randomization: $K_p, K_d \pm 15\%$.
   - Motor control latency: 1 to 2 simulation steps (20–40 ms delay buffer).
4. **Sensor Noise:**
   - IMU orientation noise: $\pm 1.5^\circ$.
   - Angular velocity noise: $\pm 0.05\text{ rad/s}$.
   - Joint encoder noise: $\pm 0.01\text{ rad}$.
5. **External Perturbations:**
   - Apply random linear force impulses to the pelvis: $F_{\text{push}} \in [0, 80\text{ N}]$ for $0.2\text{ s}$ every $3 - 5\text{ s}$.

---

### Step 4: Full Multi-Stage G1 Execution Workflow
1. **Phase 1 Training:** Run 4096 G1 environments in Isaac Gym for 5 Billion steps with CALM objective (conditional discriminator + random skill switching every $1 - 3\text{ seconds}$).
2. **Phase 2 Precision Training:** Freeze G1 low-level policy. Train high-level navigation policy with Equation (8) to track joystick heading and velocity.
3. **Phase 3 Deployment:** Export frozen low-level policy and high-level policy to ONNX / TensorRT. Run inference at 50 Hz on the G1 onboard NVIDIA Jetson Orin.
4. **Mission Autonomy:** Connect ROS 2 Nav2 / BehaviorTree to send style latents $\hat{z}$ and heading targets $d_t^*$ to the G1, achieving robust, human-like bipedal locomotion and agile loco-manipulation!
