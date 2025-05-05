# **Optimizing Computational Efficiency in Gillespie Simulation of Viral Replication**

### *Michael Sanfilippo*

The overall goal of this research project was to develop a stochastic simulation using the Gillespie algorithm to model the viral replication process of negative-sense RNA viruses. During the first research phase, the primary focus was on building a biologically accurate model that could replicate key molecular events in the viral life cycle. This included implementing essential reactions, validating the sequence of molecular events, and ensuring the system behaved realistically over simulated time.

The goal of this research phase was to optimize the Gillespie algorithm and gain a deeper understanding of the key driving forces behind the simulation. This process focused on refining the computational efficiency and accuracy of the algorithm, which simulates the stochastic dynamics of molecular interactions. The primary focus was on modeling the replication of negative-sense RNA viruses through a set of seven key molecules.

By the end of the previous research phase, we had successfully developed a stochastic model that accurately represented the viral replication process. The model was capable of simulating a five-hour replication cycle, providing a partial view of viral behavior over extended periods. However, due to computational limitations, the simulation could not run in its entirety, leaving the analysis incomplete.

Through these advancements, we gained critical insights into the algorithm's performance, particularly in terms of reaction rate sensitivity and computational scalability, setting the stage for further optimizations and analyses.

Please see the file named ***gillespie_final.ipynb*** for a full walkthrough of the optimization of our model and preliminary biological analysis.

## **References**
[1] GeeksforGeeks. (2024, April 5). Feature importance with random forests. https://www.geeksforgeeks.org/feature-importance-with-random-forests/

[2] GeeksforGeeks. (2025, April 28). Spearman correlation Heatmap in R. https://www.geeksforgeeks.org/spearman-correlation-heatmap-in-r/

[3] Gillespie, D. (n.d.). Exact stochastic simulation of coupled chemical reactions | The Journal of Physical Chemistry. https://pubs.acs.org/doi/10.1021/j100540a008

[4] Heldt, F. S. (2015, November 20). Single-cell analysis and stochastic modelling unveil large cell-to-cell variability in influenza A virus infection. Nature News. https://www.nature.com/articles/ncomms9938

[5] Numba. (n.d.-a). A ~5 minute guide to numba. A ~5 minute guide to Numba - Numba 0.52.0.dev0+274.g626b40e-py3.7-linux-x86_64.egg documentation. https://numba.pydata.org/numba-doc/dev/user/5minguide.html

[6] Numba. (n.d.-b). Just-in-time compilation. Just-in-Time compilation - Numba 0.52.0.dev0+274.g626b40e-py3.7-linux-x86_64.egg documentation. https://numba.pydata.org/numba-doc/dev/reference/jit-compilation.html
