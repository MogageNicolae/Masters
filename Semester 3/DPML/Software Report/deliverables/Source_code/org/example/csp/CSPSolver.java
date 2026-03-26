package org.example.csp;

import java.util.*;

/**
 * General Constraint Satisfaction Problem Solver.
 * Supports various search strategies (Backtracking, GBJ, Forward Checking,
 * GENET)
 * and consistency reductions (AC-3, PC-4).
 * Tracks performance metrics: nodes visited, constraint checks, backtracks, and
 * time.
 */
public class CSPSolver {
    public CSP csp;
    public long nodesVisited;
    public long constraintChecks;
    public long backtrackCount;
    public long startTime;

    private final List<Reduction> reductions = new ArrayList<>();
    private SearchStrategy searchStrategy;

    public CSPSolver(CSP csp) {
        this.csp = csp;
    }

    public CSPSolver addReduction(Reduction reduction) {
        this.reductions.add(reduction);
        return this;
    }

    public CSPSolver setSearch(SearchStrategy strategy) {
        this.searchStrategy = strategy;
        return this;
    }

    /**
     * Solves the CSP using the configured reductions and search strategy.
     * 
     * @return true if a solution is found, false otherwise.
     */
    public boolean solve() {
        resetMetrics();
        for (Reduction r : reductions) {
            if (!r.reduce(this)) {
                return false; // Domain wipeout
            }
        }
        if (searchStrategy != null) {
            return searchStrategy.solve(this);
        }
        return true; // No search strategy, just reductions
    }

    // Set of pruned pairs (Constraint -> Set of disallowed value pairs)
    private final Map<Constraint, Set<String>> prunedPairs = new HashMap<>();

    /**
     * Resets all metrics and variable assignments.
     */
    public void resetMetrics() {
        nodesVisited = 0;
        constraintChecks = 0;
        backtrackCount = 0;
        startTime = System.currentTimeMillis();
        prunedPairs.clear();
        // Reset assignments
        for (Variable v : csp.variables)
            v.assignedValue = null;
    }

    public void incrementNodesVisited() {
        nodesVisited++;
    }

    public void incrementConstraintChecks() {
        constraintChecks++;
    }

    /**
     * Helper to check if pair is allowed (considering PC-4 pruning).
     * 
     * @param c  The constraint.
     * @param i  Index of first variable.
     * @param vi Value of first variable.
     * @param j  Index of second variable.
     * @param vj Value of second variable.
     * @return true if allowed, false otherwise.
     */
    public boolean isPairAllowed(Constraint c, int i, Object vi, int j, Object vj) {
        // Check base constraint
        boolean satisfied = (c.var1Idx == i) ? c.isSatisfied(vi, vj) : c.isSatisfied(vj, vi);
        if (!satisfied)
            return false;

        // Check pruned pairs from PC-4
        if (prunedPairs.containsKey(c)) {
            String key = (c.var1Idx == i) ? (vi + ":" + vj) : (vj + ":" + vi);
            return !prunedPairs.get(c).contains(key);
        }
        return true;
    }

    /**
     * Prunes a pair of values for a constraint (used by PC-4).
     */
    public void prunePair(Constraint c, int i, Object vi, int j, Object vj) {
        prunedPairs.putIfAbsent(c, new HashSet<>());
        String key = (c.var1Idx == i) ? (vi + ":" + vj) : (vj + ":" + vi);
        prunedPairs.get(c).add(key);
    }

    public Map<Constraint, Set<String>> getPrunedPairs() {
        return prunedPairs;
    }

    public void setPrunedPairs(Map<Constraint, Set<String>> pairs) {
        this.prunedPairs.clear();
        this.prunedPairs.putAll(pairs);
    }

    /**
     * Checks if the current assignment of 'var' is consistent with all
     * previously assigned neighbors.
     * 
     * @param var The variable to check.
     * @return true if consistent, false otherwise.
     */
    public boolean isConsistent(Variable var) {
        for (Constraint c : csp.adjMap.get(var.id)) {
            Variable neighbor = csp.variables.get((c.var1Idx == var.id) ? c.var2Idx : c.var1Idx);
            // Check consistency with already assigned neighbors
            if (neighbor.assignedValue != null) {
                constraintChecks++;
                boolean ok = (c.var1Idx == var.id)
                        ? c.isSatisfied(var.assignedValue, neighbor.assignedValue)
                        : c.isSatisfied(neighbor.assignedValue, var.assignedValue);
                if (!ok)
                    return false;
            }
        }
        return true;
    }

    /**
     * Finds the minimum index of a variable that conflicts with the current
     * assignment.
     * Used by GBJ.
     * 
     * @param var        The variable to check.
     * @param currentIdx The current index in the search.
     * @return The index of the conflicting variable, or -1 if consistent.
     */
    public int checkConsistencyAndGetMinConflict(Variable var, int currentIdx) {
        int minConflict = -1;

        for (Constraint c : csp.adjMap.get(var.id)) {
            int neighborIdx = (c.var1Idx == var.id) ? c.var2Idx : c.var1Idx;
            if (neighborIdx < currentIdx) {
                Variable neighbor = csp.variables.get(neighborIdx);
                // Only check if neighbor is assigned (it should be, since index < currentIdx)
                if (neighbor.assignedValue != null) {
                    constraintChecks++;
                    boolean ok = (c.var1Idx == var.id)
                            ? c.isSatisfied(var.assignedValue, neighbor.assignedValue)
                            : c.isSatisfied(neighbor.assignedValue, var.assignedValue);

                    if (!ok) {
                        if (minConflict == -1 || neighborIdx < minConflict) {
                            minConflict = neighborIdx;
                        }
                    }
                }
            }
        }
        return minConflict;
    }
}
