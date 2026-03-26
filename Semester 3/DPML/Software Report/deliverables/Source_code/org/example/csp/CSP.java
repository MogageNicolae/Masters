package org.example.csp;

import java.util.*;

/**
 * Represents a Constraint Satisfaction Problem.
 * Consists of variables, their domains, and constraints between them.
 */
public class CSP {
    public List<Variable> variables = new ArrayList<>();
    public List<Constraint> constraints = new ArrayList<>();
    public Map<Integer, List<Constraint>> adjMap = new HashMap<>(); // Adjacency list for fast lookup

    /**
     * Adds a variable to the CSP.
     * 
     * @param v The variable to add.
     */
    public void addVariable(Variable v) {
        variables.add(v);
        adjMap.put(v.id, new ArrayList<>());
    }

    /**
     * Adds a constraint to the CSP and updates the adjacency map.
     * 
     * @param c The constraint to add.
     */
    public void addConstraint(Constraint c) {
        constraints.add(c);
        adjMap.get(c.var1Idx).add(c);
        adjMap.get(c.var2Idx).add(c);
    }

    /**
     * Prints metrics about the constraint graph (nodes, edges, density).
     */
    public void printGraphMetrics() {
        System.out.println("=== Constraint Graph Metrics ===");
        System.out.println("Variables (Nodes): " + variables.size());
        System.out.println("Constraints (Edges): " + constraints.size());
        double density = (variables.size() > 1)
                ? (2.0 * constraints.size()) / (variables.size() * (variables.size() - 1))
                : 0;
        System.out.printf("Graph Density: %.4f\n", density);
        System.out.println("--------------------------------");
    }

    /**
     * Helper to get binary constraint between two vars if exists.
     * 
     * @param i Index of first variable.
     * @param j Index of second variable.
     * @return The constraint if found, null otherwise.
     */
    public Constraint getConstraint(int i, int j) {
        for (Constraint c : adjMap.get(i)) {
            if ((c.var1Idx == i && c.var2Idx == j) || (c.var1Idx == j && c.var2Idx == i))
                return c;
        }
        return null;
    }
}
