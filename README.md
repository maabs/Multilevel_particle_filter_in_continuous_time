# Multilevel_particle_filter_in_continuous_time
Multilevel methodology applied to the continuous particle filter

This code takes the particle filter and creates coupled versions at subsequent time discretization levels. It ties them together using multilevel methodology. The coupling is both is the brownian motions to propagate the particles and in the resampling step. 

The code is composed of several functions concluding with the function MLPF that computes tthe MLPF.
