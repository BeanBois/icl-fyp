
"""
Idea 1: Projecting Actions and Local Graph into a Feature space s.t addition/subtraction can occur
    - Basically we want to framework where G(t) + A(t) = G(t+1) and G(t+1) - G(t) = A(t) (too similar to Q learning, think about this! )
    - we can imagine a demo sequence to be a simultaneous equation (interesting thought eleaborate)
        - So G(0) + A(0) + G(1) ... + A(L-1) = G(L) 
        - Kinda imagine it like a more complicated SAT problem 
            - Every generated Pseudo-seq can be thought of to be semantically consistent
                * Thus it is possible to simulate ‘arbitrary but consistent’ trajectories as training data. Here, consistent means that while the trajectories differ, 
                they ‘perform’ the same type of pseudo-task at a semantic level. -- paper
                * or more specifically:
                These pseudo-demonstrations don't need to be:
                    Kinematically feasible (impossible robot movements are OK)
                    Dynamically feasible (ignoring physics is OK)  
                    Optimal paths (inefficient routes are OK)
                    Human-demonstrated (no real human needed)
                All pseudo-demonstrations for a pseudo-task should:
                    ✅ Show the SAME type of behavior pattern
                    ✅ Have the SAME goal structure
                    ✅ Follow the SAME semantic meaning
    - So assuming that all generated demostrations follows similar semantics{ Eg: sub-task1 -> subtask2 -> ... }
        - we want an agent that holds an unbiased/arbitarary semantics knowledge (pre-given_demo)
    - Also assume that G(t) and A(t) in this semantic space   
        - PC embedding works fine to capture this space
        - Firstly imagine a general underlying sequence of semantic (Hidden Markov Chain) that occurs when a demo is performed 
            - we want to project our current graph sequence into this HMC states with PC (Diffusion model to learn this)
                * mapping is a many-to-many! 
            - 
    - BIG VIEW:
        Given that G(t) => LG(t) and G(t) + A(t) => AG(t) / LG'(t) -- following LocalGraph and ActionGraph from IP 
        From PseudoDemo we get:
            LG(0), LG'(0), LG(1), LG'(1) .... LG(L-1), LG'(L-1), G(L) # only taking agent nodes here 
        Then assume underlying Semnatic HMC is 
            F(0), FA(0), F(1) ... F(M-1), FA(M-1), F(M). where F(m) represents semantic stage, FA(m) represents semantic action 
            Theorectically LG(.) maps to F(.) and LG'(.) maps to FA(.) # Actor Critic here? (At the same time LG'(t) & LG(t) should be similar hmm think abt this)
                * Here 2 models V(.) which learns underlying r/s for LG(.) F(.), and Q(.) which learns r/s for LG'(.) and FA(.)
                    - V(.) will be critic and be used only during training 
                        * this is so as it is more computationally intensive (require scene nodes and all that)
                        * can even take this further with 1 (Critic) using a more expressive embedding than the other (Actor)
                    - Q(.) will be the actor
                        * Only require modelling of the agent nodes

        AIM RECAP: (1) SEMANTICALLY TRAIN AN AGENT (2) FIND A METHOD THAT ALLOWS GIVEN DEMONSTRATIONS TO SKE SEMANTICS OF AGENT

        (1) SEMANTICALLY TRAIN AN AGENT 
            By learning the underlying Semantic Chain, or mathematically:
                - Given G(t) {point clouds and agent pos and agent state}, 
                - and that we can perform operations : G(T) => LG(T)  &   A(T), LG(T) => LG'(T) 
                - we are able to map LG'(T) to a semantic state F(M) (Encoder)

            * Eg semantics for us : Move gripper towards object, Move Around object, 


"""