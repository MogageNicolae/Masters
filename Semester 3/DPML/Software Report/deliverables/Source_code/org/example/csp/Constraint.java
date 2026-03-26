package org.example.csp;

/**
 * Abstract base class for binary constraints.
 * Defines a relationship between two variables that must be satisfied.
 */
public abstract class Constraint {
    public int var1Idx, var2Idx;
    public String description = "";

    public Constraint(int v1, int v2) {
        this.var1Idx = v1;
        this.var2Idx = v2;
    }

    public Constraint(int v1, int v2, String description) {
        this.var1Idx = v1;
        this.var2Idx = v2;
        this.description = description;
    }

    /**
     * Checks if the pair of values satisfies the constraint.
     * 
     * @param val1 Value assigned to the first variable (var1Idx).
     * @param val2 Value assigned to the second variable (var2Idx).
     * @return true if satisfied, false otherwise.
     */
    public abstract boolean isSatisfied(Object val1, Object val2);
}
