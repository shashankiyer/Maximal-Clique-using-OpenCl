# Maximal-Clique-using-OpenCl

This is parallel algorithm for finding maximal cliques in a graph. A clique is a set of vertices C in a graph G, such that for every i,j∈C, the edge {i,j} is in G. Such a clique is maximal if it is not a proper subset of any other clique. Given a graph G=(V,E), the algorithm computes its dual Gbar=(V,Ebar), where Ebar={(i,j)|(i,j)∈V×V,(i,j)∉E}. A maximal set is a clique in G if and only if it is as an independent set in Gbar.

The code accepts as input an adjacency matrix and computes the maximal clique using OpenCl. 

It outputs an array indicating the groups of vertices that form a maximal clique.
