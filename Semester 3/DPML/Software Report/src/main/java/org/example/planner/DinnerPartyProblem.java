package org.example.planner;

import org.example.csp.*;
import java.util.*;

public class DinnerPartyProblem {

    /**
     * Creates a solvable Dinner Party CSP by first generating a valid solution,
     * then adding constraints that are compatible with that solution.
     * Constraint types:
     * 1. Enemy constraints: certain pairs must sit at different tables
     * 2. Partner constraints: certain pairs must sit at the same table
     * 3. Political balance: guests with similar political scores (within threshold)
     * prefer to sit together; those with very different scores prefer separation
     * 
     * @param numGuests           Number of guests to seat
     * @param numTables           Number of tables available
     * @param tableCapacity       Maximum capacity per table
     * @param enemyProbability    Probability (0.0-1.0) of adding enemy constraints
     *                            from available pairs
     * @param partnerProbability  Probability (0.0-1.0) of adding partner
     *                            constraints from available pairs
     * @param politicsProbability Probability (0.0-1.0) of adding political balance
     *                            constraints
     * @return A CSP instance with variables and constraints
     */
    public static CSP create(int numGuests, int numTables, int tableCapacity,
            double enemyProbability, double partnerProbability,
            double politicsProbability) {
        // Validate inputs
        if (numGuests > numTables * tableCapacity) {
            throw new IllegalArgumentException(
                    "Not enough table capacity: " + numGuests + " guests need " +
                            (numGuests / tableCapacity + 1) + " tables of capacity " + tableCapacity);
        }

        CSP csp = new CSP();
        Random rand = new Random(42); // Seed for reproducibility

        // Assign political scores to each guest (scale 1-10)
        int[] politicalScores = new int[numGuests];
        for (int i = 0; i < numGuests; i++) {
            politicalScores[i] = 1 + rand.nextInt(10); // Score from 1 to 10
        }

        // Political balance threshold
        int politicalThreshold = 2; // Guests with score difference > 2 should sit apart

        // Step 1: Generate a valid solution first
        int[] validSolution = generateValidSolution(numGuests, numTables, tableCapacity,
                politicalScores, rand);

        // Step 2: Create variables
        List<Object> tables = new ArrayList<>();
        for (int i = 1; i <= numTables; i++) {
            tables.add(i);
        }

        for (int i = 0; i < numGuests; i++) {
            csp.addVariable(new Variable(i, "Guest" + i, tables));
        }

        // Step 3: Generate constraints compatible with the solution
        Set<String> sameTablePairs = new HashSet<>();
        Set<String> diffTablePairs = new HashSet<>();

        for (int i = 0; i < numGuests; i++) {
            for (int j = i + 1; j < numGuests; j++) {
                String pairKey = i + "," + j;
                if (validSolution[i] == validSolution[j]) {
                    sameTablePairs.add(pairKey);
                } else {
                    diffTablePairs.add(pairKey);
                }
            }
        }

        // Track existing constraints to avoid duplicates
        Set<String> existingConstraints = new HashSet<>();

        // Add "Enemy" constraints (must be at different tables)
        List<String> diffPairsList = new ArrayList<>(diffTablePairs);
        Collections.shuffle(diffPairsList, rand);
        int numEnemyPairs = (int) (diffPairsList.size() * enemyProbability);

        for (int k = 0; k < numEnemyPairs; k++) {
            String[] parts = diffPairsList.get(k).split(",");
            int i = Integer.parseInt(parts[0]);
            int j = Integer.parseInt(parts[1]);

            csp.addConstraint(new Constraint(i, j, "Enemy") {
                public boolean isSatisfied(Object v1, Object v2) {
                    return !v1.equals(v2); // Different tables (enemies)
                }
            });
            existingConstraints.add(i + "," + j);
        }

        // Add "Partner" constraints (must be at same table)
        List<String> samePairsList = new ArrayList<>(sameTablePairs);
        Collections.shuffle(samePairsList, rand);
        int numPartnerPairs = (int) (samePairsList.size() * partnerProbability);

        for (int k = 0; k < numPartnerPairs; k++) {
            String[] parts = samePairsList.get(k).split(",");
            int i = Integer.parseInt(parts[0]);
            int j = Integer.parseInt(parts[1]);

            csp.addConstraint(new Constraint(i, j, "Partner") {
                public boolean isSatisfied(Object v1, Object v2) {
                    return v1.equals(v2); // Same table (partners)
                }
            });
            existingConstraints.add(i + "," + j);
        }

        // Add "Political Balance" constraints
        // If |P_X - P_Y| <= threshold, they prefer same table
        // If |P_X - P_Y| > threshold, they prefer different tables
        for (int i = 0; i < numGuests; i++) {
            for (int j = i + 1; j < numGuests; j++) {
                // Skip if a constraint already exists for this pair
                if (existingConstraints.contains(i + "," + j)) {
                    continue;
                }

                int scoreDiff = Math.abs(politicalScores[i] - politicalScores[j]);

                // Add constraint based on probability
                if (rand.nextDouble() < politicsProbability) {
                    if (scoreDiff <= politicalThreshold) {
                        // Similar political views - prefer same table
                        // Only add if they're actually at the same table in the solution
                        if (validSolution[i] == validSolution[j]) {
                            csp.addConstraint(new Constraint(i, j, "Political Ally") {
                                public boolean isSatisfied(Object v1, Object v2) {
                                    return v1.equals(v2); // Same table (political allies)
                                }
                            });
                            existingConstraints.add(i + "," + j);
                        }
                    } else {
                        // Very different political views - prefer different tables
                        // Only add if they're actually at different tables in the solution
                        if (validSolution[i] != validSolution[j]) {
                            csp.addConstraint(new Constraint(i, j, "Political Opponent") {
                                public boolean isSatisfied(Object v1, Object v2) {
                                    return !v1.equals(v2); // Different tables (political opponents)
                                }
                            });
                            existingConstraints.add(i + "," + j);
                        }
                    }
                }
            }
        }
        return csp;
    }

    /**
     * Generates a valid table assignment respecting capacity constraints
     * and trying to group guests with similar political scores.
     */
    private static int[] generateValidSolution(int numGuests, int numTables, int tableCapacity,
            int[] politicalScores, Random rand) {
        int[] solution = new int[numGuests];
        int[] tableOccupancy = new int[numTables + 1]; // 1-indexed

        // Create a list of guests sorted by political score to help clustering
        List<Integer> guestList = new ArrayList<>();
        for (int i = 0; i < numGuests; i++) {
            guestList.add(i);
        }

        // Sort by political score to encourage similar-minded guests at same tables
        guestList.sort(Comparator.comparingInt(i -> politicalScores[i]));

        // Distribute guests trying to keep similar political scores together
        int currentTable = 1;
        for (int guest : guestList) {
            // If current table is full, move to next table
            if (tableOccupancy[currentTable] >= tableCapacity) {
                currentTable++;
                if (currentTable > numTables) {
                    // Need to backtrack - shouldn't happen with valid inputs
                    throw new IllegalStateException("Ran out of tables during solution generation");
                }
            }

            // Assign guest to current table
            solution[guest] = currentTable;
            tableOccupancy[currentTable]++;
        }

        // Add some randomness to avoid perfect clustering
        // Swap some guests between tables
        for (int i = 0; i < numGuests / 4; i++) {
            int guest1 = rand.nextInt(numGuests);
            int guest2 = rand.nextInt(numGuests);

            if (guest1 != guest2) {
                // Swap their table assignments
                int temp = solution[guest1];
                solution[guest1] = solution[guest2];
                solution[guest2] = temp;
            }
        }

        return solution;
    }
}
