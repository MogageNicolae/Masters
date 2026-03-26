package org.example.planner;

import org.example.csp.CSP;
import org.example.csp.Constraint;
import org.example.csp.Variable;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * Renders and saves Dinner Party problem solutions.
 */
public class DinnerPartyRenderer {

    /**
     * Saves the constraint distribution and solution for a configuration.
     *
     * @param csp       The solved CSP containing the solution
     * @param outputDir The directory where the output file should be saved
     * @param label     The configuration label (used for filename)
     * @throws IOException if file writing fails
     */
    public static void saveConstraintDistribution(CSP csp, String outputDir, String label) throws IOException {
        String filename = outputDir + "/" + label.toLowerCase() + "_solution.txt";

        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("# Dinner Party Problem - " + label + " Configuration");
            writer.println("=".repeat(60));
            writer.println();

            // Solution - Table Assignments
            writer.println("## Solution - Table Assignments");
            writer.println("-".repeat(60));

            // Group guests by table
            Map<Object, List<Variable>> tableAssignments = new java.util.HashMap<>();
            for (Variable v : csp.variables) {
                if (v.assignedValue != null) {
                    tableAssignments.computeIfAbsent(v.assignedValue, k -> new java.util.ArrayList<>()).add(v);
                }
            }

            // Print each table
            List<Object> tables = new java.util.ArrayList<>(tableAssignments.keySet());
            tables.sort(Comparator.comparing(a -> ((Integer) a)));

            for (Object table : tables) {
                List<Variable> guests = tableAssignments.get(table);
                writer.println(String.format("Table %s (%d guests):", table, guests.size()));
                for (Variable guest : guests) {
                    writer.println("  - " + guest.name);
                }
                writer.println();
            }

            // Guest Constraint Details
            writer.println("## Guest Constraint Details");
            writer.println("-".repeat(60));

            for (int i = 0; i < csp.variables.size(); i++) {
                Variable guest = csp.variables.get(i);
                writer.println(String.format("%s (Table %s):", guest.name,
                        guest.assignedValue != null ? guest.assignedValue : "UNASSIGNED"));

                // Find all constraints involving this guest
                List<String> enemies = new java.util.ArrayList<>();
                List<String> partners = new java.util.ArrayList<>();

                for (Constraint c : csp.constraints) {
                    if (c.var1Idx == i || c.var2Idx == i) {
                        int otherIdx = (c.var1Idx == i) ? c.var2Idx : c.var1Idx;
                        Variable other = csp.variables.get(otherIdx);

                        switch (c.description) {
                            case "Enemy" -> enemies.add(other.name);
                            case "Partner" -> partners.add(other.name);
                            case "Political Ally" -> partners.add(other.name + " (Political)");
                            case "Political Opponent" -> enemies.add(other.name + " (Political)");
                            default -> {
                                if (guest.domain.size() >= 2) {
                                    Object val1 = guest.domain.get(0);
                                    Object val2 = guest.domain.get(1);

                                    boolean allowsSame = c.isSatisfied(val1, val1);
                                    boolean allowsDiff = c.isSatisfied(val1, val2);

                                    if (allowsSame && !allowsDiff) {
                                        partners.add(other.name);
                                    } else if (!allowsSame && allowsDiff) {
                                        enemies.add(other.name);
                                    }
                                }
                            }
                        }
                    }
                }

                if (!enemies.isEmpty()) {
                    writer.println("  Enemies/Different-Table: " + String.join(", ", enemies));
                }
                if (!partners.isEmpty()) {
                    writer.println("  Partners/Same-Table: " + String.join(", ", partners));
                }
                if (enemies.isEmpty() && partners.isEmpty()) {
                    writer.println("  No constraints");
                }
                writer.println();
            }
        }
    }
}
