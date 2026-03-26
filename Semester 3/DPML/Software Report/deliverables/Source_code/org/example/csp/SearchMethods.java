package org.example.csp;

import java.util.*;

public class SearchMethods {

    /**
     * Standard Backtracking search.
     * Assigns values to variables one by one and checks consistency.
     * If a conflict is found, it backtracks to the previous variable.
     */
    public static class Backtracking implements SearchStrategy {
        @Override
        public boolean solve(CSPSolver solver) {
            return btRecursive(solver, 0);
        }

        private boolean btRecursive(CSPSolver solver, int idx) {
            if (idx == solver.csp.variables.size())
                return true;

            Variable var = solver.csp.variables.get(idx);
            solver.incrementNodesVisited();

            for (Object val : var.domain) {
                var.assignedValue = val;
                if (solver.isConsistent(var)) {
                    if (btRecursive(solver, idx + 1))
                        return true;
                    solver.backtrackCount++; // Increment backtrack counter
                }
            }
            var.assignedValue = null;
            return false;
        }
    }

    /**
     * Graph-Based Backjumping (GBJ).
     * Maintains a "conflict set" (ancestors) for each variable.
     * When a dead-end is reached, it jumps back to the most recent ancestor
     * that caused the conflict, rather than just the immediate predecessor.
     */
    public static class GBJ implements SearchStrategy {
        @Override
        public boolean solve(CSPSolver solver) {
            // Initialize ancestral info
            for (int i = 0; i < solver.csp.variables.size(); i++)
                solver.csp.variables.get(i).ancestors.clear();
            return gbjRecursive(solver, 0) == -2;
        }

        private int gbjRecursive(CSPSolver solver, int idx) {
            if (idx == solver.csp.variables.size())
                return -2; // Solution found

            Variable var = solver.csp.variables.get(idx);
            solver.incrementNodesVisited();

            for (Object val : var.domain) {
                var.assignedValue = val;

                int conflictIdx = solver.checkConsistencyAndGetMinConflict(var, idx);

                if (conflictIdx == -1) { // Consistent
                    int result = gbjRecursive(solver, idx + 1);
                    if (result == -2)
                        return -2; // Success

                    if (result < idx) {
                        var.assignedValue = null;
                        solver.backtrackCount++; // Increment backtrack counter
                        return result;
                    }
                    solver.backtrackCount++; // Backjump occurred
                } else {
                    var.ancestors.add(conflictIdx);
                }
            }

            var.assignedValue = null;

            if (var.ancestors.isEmpty()) {
                return idx - 1;
            }

            int jumpTo = -1;
            for (int anc : var.ancestors) {
                if (anc > jumpTo)
                    jumpTo = anc;
            }

            if (jumpTo > -1) {
                solver.csp.variables.get(jumpTo).ancestors.addAll(var.ancestors);
                solver.csp.variables.get(jumpTo).ancestors.remove(jumpTo);
            }

            return jumpTo;
        }
    }

    /**
     * Forward Checking (FC).
     * When a variable is assigned, it filters the domains of all future unassigned
     * variables
     * connected by constraints. If any future domain becomes empty, it backtracks
     * immediately.
     */
    public static class ForwardChecking implements SearchStrategy {
        @Override
        public boolean solve(CSPSolver solver) {
            return fcRecursive(solver, 0);
        }

        private boolean fcRecursive(CSPSolver solver, int idx) {
            if (idx == solver.csp.variables.size())
                return true;

            Variable var = solver.csp.variables.get(idx);
            solver.incrementNodesVisited();

            List<Object> currentDomain = new ArrayList<>(var.domain);

            for (Object val : currentDomain) {
                var.assignedValue = val;

                Map<Integer, List<Object>> savedDomains = new HashMap<>();
                boolean wipeout = false;

                for (int i = idx + 1; i < solver.csp.variables.size(); i++) {
                    Variable futureVar = solver.csp.variables.get(i);
                    Constraint c = solver.csp.getConstraint(var.id, futureVar.id);

                    if (c != null) {
                        List<Object> prunedDomain = new ArrayList<>();
                        boolean changed = false;

                        for (Object futureVal : futureVar.domain) {
                            solver.incrementConstraintChecks();
                            if (solver.isPairAllowed(c, var.id, val, futureVar.id, futureVal)) {
                                prunedDomain.add(futureVal);
                            } else {
                                changed = true;
                            }
                        }

                        if (changed) {
                            if (!savedDomains.containsKey(futureVar.id)) {
                                savedDomains.put(futureVar.id, new ArrayList<>(futureVar.domain));
                            }
                            futureVar.domain = prunedDomain;

                            if (futureVar.domain.isEmpty()) {
                                wipeout = true;
                                break;
                            }
                        }
                    }
                }

                if (!wipeout) {
                    if (fcRecursive(solver, idx + 1))
                        return true;
                    solver.backtrackCount++; // Increment backtrack counter
                }

                for (Map.Entry<Integer, List<Object>> entry : savedDomains.entrySet()) {
                    solver.csp.variables.get(entry.getKey()).domain = entry.getValue();
                }
            }

            var.assignedValue = null;
            return false;
        }
    }

    /**
     * GENET (General Neural Network-based local search).
     * A stochastic local search algorithm that uses a min-conflict heuristic
     * with dynamic edge weighting to escape local minima.
     * 
     * 1. Initialize variables randomly.
     * 2. Iteratively improve assignments to minimize constraint violations (cost).
     * 3. If stuck in a local minimum, increase weights of violated constraints.
     */
    public static class GENET implements SearchStrategy {
        private final int maxCycles;

        public GENET(int maxCycles) {
            this.maxCycles = maxCycles;
        }

        @Override
        public boolean solve(CSPSolver solver) {
            Map<Constraint, Integer> weights = new HashMap<>();
            for (Constraint c : solver.csp.constraints)
                weights.put(c, 1);

            Random rand = new Random();
            for (Variable v : solver.csp.variables) {
                if (v.domain.isEmpty())
                    return false;
                v.assignedValue = v.domain.get(rand.nextInt(v.domain.size()));
            }

            for (int cycle = 0; cycle < maxCycles; cycle++) {
                solver.incrementNodesVisited();

                int totalCost = 0;
                List<Constraint> violated = new ArrayList<>();
                for (Constraint c : solver.csp.constraints) {
                    Variable v1 = solver.csp.variables.get(c.var1Idx);
                    Variable v2 = solver.csp.variables.get(c.var2Idx);
                    solver.incrementConstraintChecks();
                    if (!c.isSatisfied(v1.assignedValue, v2.assignedValue)) {
                        totalCost += weights.get(c);
                        violated.add(c);
                    }
                }

                if (totalCost == 0)
                    return true;

                boolean localMinimum = true;

                for (Variable v : solver.csp.variables) {
                    Object bestVal = v.assignedValue;
                    int minViolation = Integer.MAX_VALUE;

                    for (Object val : v.domain) {
                        int currentViolation = 0;
                        for (Constraint c : solver.csp.adjMap.get(v.id)) {
                            Variable neighbor = solver.csp.variables.get((c.var1Idx == v.id) ? c.var2Idx : c.var1Idx);
                            if (!c.isSatisfied((c.var1Idx == v.id) ? val : neighbor.assignedValue,
                                    (c.var1Idx == v.id) ? neighbor.assignedValue : val)) {
                                currentViolation += weights.get(c);
                            }
                        }

                        if (currentViolation < minViolation) {
                            minViolation = currentViolation;
                            bestVal = val;
                        }
                    }

                    if (minViolation < totalCost && !bestVal.equals(v.assignedValue)) {
                        v.assignedValue = bestVal;
                        localMinimum = false;
                    }
                }

                if (localMinimum) {
                    for (Constraint c : violated) {
                        weights.put(c, weights.get(c) + 1);
                    }
                }
            }
            return false;
        }
    }
}
