## PAT-Net (Part-independent Attention Network for Skeleton Based Human Action Recognition)
The self-attention in the Transformer presented a pioneering approach in modeling global long-range spatio-temporal relationships through a dot-product operation.
However, the attention maps for different queries in weight computation remain almost the same, and only a query-independent relationship is learned.
These observations indicate the existence of both pairwise and unary relationship influences in the self-attention weight computation in a tightly coupled form, which prevents intra-frame and inter-frame action feature extraction. 
To address these issues, we design a Part-independent  Attention Network (PAT-Net) for skeleton-based human action recognition. The PAT-Net contains a whitened pairwise self-attention, unary self-attention and position attention as independent functions and different projection matrices for learning representative action features. 
The whitened pairwise self-attention captures the influence of a single key joint specifically on another query joint, and the unary self-attention models the general impact of one key joint over all other query joints to learn the discriminative action features. Furthermore, we design a position attention learning module that computes the correlation between  action semantics and position information separately with different projection matrices. 

# Architecture of PAT-Net
!
