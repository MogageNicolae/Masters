package org.example.csp;

import java.util.*;

public class Reductions {

    /**
     * Arc Consistency 3 (AC-3).
     * Iteratively removes values from domains that have no support in neighboring
     * domains.
     * Uses a queue to propagate constraints until a fixed point is reached.
     */
    public static class AC3 implements Reduction {
        @Override
        public boolean reduce(CSPSolver solver) {
            Queue<Constraint> queue = new LinkedList<>(solver.csp.constraints);

            while (!queue.isEmpty()) {
                Constraint c = queue.poll();
                if (revise(solver, c.var1Idx, c.var2Idx, c)) {
                    if (solver.csp.variables.get(c.var1Idx).domainSize() == 0)
                        return false;
                    for (Constraint neighbor : solver.csp.adjMap.get(c.var1Idx)) {
                        if (neighbor != c)
                            queue.add(neighbor);
                    }
                }
                if (revise(solver, c.var2Idx, c.var1Idx, c)) {
                    if (solver.csp.variables.get(c.var2Idx).domainSize() == 0)
                        return false;
                    for (Constraint neighbor : solver.csp.adjMap.get(c.var2Idx)) {
                        if (neighbor != c)
                            queue.add(neighbor);
                    }
                }
            }
            return true;
        }

        private boolean revise(CSPSolver solver, int i, int j, Constraint c) {
            boolean revised = false;
            Variable Xi = solver.csp.variables.get(i);
            Variable Xj = solver.csp.variables.get(j);

            Iterator<Object> it = Xi.domain.iterator();
            while (it.hasNext()) {
                Object x = it.next();
                boolean supported = false;
                for (Object y : Xj.domain) {
                    solver.incrementConstraintChecks();
                    if (solver.isPairAllowed(c, i, x, j, y)) {
                        supported = true;
                        break;
                    }
                }
                if (!supported) {
                    it.remove();
                    revised = true;
                }
            }
            return revised;
        }
    }

    /**
     * Path Consistency 4 (PC-4).
     * Enforces path consistency by pruning pairs of values that cannot be extended
     * to a third variable.
     *
     * Implementation based strictly on lecture notes.
     * WARNING: High space complexity O(n^3 a^3).
     */
    public static class PC4 implements Reduction {

        // Represents a specific assignment (Variable Index, Value)
        record Label(int varIdx, Object val) {}

        // Represents a pair of assignments (Variable i, Value b) - (Variable j, Value c)
        record CompoundLabel(int i, Object b, int j, Object c) {}

        // S: Map<Supporting_Pair, Set<Supported_Pair_End>>
        // Meaning: If Key(i,b,k,d) is removed, check Target(i,b,j,c)
        // We only store the (j,c) part in the value set to save space,
        // implying the 'i,b' part is shared with the key.
        private Map<CompoundLabel, List<Label>> S;

        // Counter: Use int[] instead of HashMap<Integer, Integer>
        private Map<CompoundLabel, int[]> Counter;

        private Set<CompoundLabel> M;
        private Queue<CompoundLabel> LIST;

        @Override
        public boolean reduce(CSPSolver solver) {
            S = new HashMap<>();
            Counter = new HashMap<>();
            M = new HashSet<>();
            LIST = new LinkedList<>();

            int n = solver.csp.variables.size();

            // -------------------------------------------------
            // Step 1: Initialization
            // -------------------------------------------------
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i == j) continue;

                    // We only check pairs that are constrained (or implicitly allowed if null)
                    // PC4 usually runs on a complete graph, but we check constraints existence.
                    Constraint Cij = solver.csp.getConstraint(i, j);
                    if (Cij == null) continue; // Skip if no direct constraint to prune

                    Variable Vi = solver.csp.variables.get(i);
                    Variable Vj = solver.csp.variables.get(j);

                    for (Object b : Vi.domain) {
                        for (Object c : Vj.domain) {

                            // If the pair itself is invalid by Cij, it's already pruned or we prune now
                            if (!solver.isPairAllowed(Cij, i, b, j, c)) continue;

                            CompoundLabel currentPair = new CompoundLabel(i, b, j, c);
                            int[] counts = new int[n];
                            Counter.put(currentPair, counts);

                            // Check support from every variable k
                            for (int k = 0; k < n; k++) {
                                if (k == i || k == j) continue;

                                int totalSupportFromK = 0;
                                Variable Vk = solver.csp.variables.get(k);
                                Constraint Cik = solver.csp.getConstraint(i, k);
                                Constraint Ckj = solver.csp.getConstraint(k, j);

                                // If there are no constraints to k, it implicitly supports everything.
                                // But assuming we only care about existing triangles:
                                if (Cik != null && Ckj != null) {
                                    for (Object d : Vk.domain) {
                                        boolean support1 = solver.isPairAllowed(Cik, i, b, k, d);
                                        boolean support2 = solver.isPairAllowed(Ckj, k, d, j, c); // Note direction k->j

                                        if (support1 && support2) {
                                            totalSupportFromK++;

                                            CompoundLabel supportingEdge = new CompoundLabel(i, b, k, d);

                                            // Use ArrayList (computeIfAbsent is fine)
                                            S.computeIfAbsent(supportingEdge, x -> new ArrayList<>()).add(new Label(j, c));
                                        }
                                    }

                                    counts[k] = totalSupportFromK;

                                    if (totalSupportFromK == 0) {
                                        reject(solver, currentPair);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // -------------------------------------------------
            // Step 2: Propagation
            // -------------------------------------------------
            while (!LIST.isEmpty()) {
                CompoundLabel rejected = LIST.poll();

                if (S.containsKey(rejected)) {
                    for (Label supportedEnd : S.get(rejected)) {

                        int i = rejected.i();
                        Object b = rejected.b();
                        int k = rejected.j();

                        int j = supportedEnd.varIdx();
                        Object c = supportedEnd.val();

                        CompoundLabel supportedPair = new CompoundLabel(i, b, j, c);

                        int[] countsToCheck = Counter.get(supportedPair);

                        if (countsToCheck != null) {
                            countsToCheck[k]--;

                            if (countsToCheck[k] == 0 && !M.contains(supportedPair)) {
                                reject(solver, supportedPair);
                            }
                        }
                    }
                }
            }

            return true;
        }

        private void reject(CSPSolver solver, CompoundLabel cl) {
            if (M.contains(cl)) return;

            M.add(cl);
            LIST.add(cl);

            Constraint C = solver.csp.getConstraint(cl.i(), cl.j());
            if (C != null) {
                solver.prunePair(C, cl.i(), cl.b(), cl.j(), cl.c());
            }

            CompoundLabel sym = new CompoundLabel(cl.j(), cl.c(), cl.i(), cl.b());
            if (!M.contains(sym)) {
                M.add(sym);
                LIST.add(sym);
            }
        }
    }
}
