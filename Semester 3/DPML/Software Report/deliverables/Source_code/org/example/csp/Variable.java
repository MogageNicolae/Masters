package org.example.csp;

import java.util.*;

/**
 * Represents a variable in the CSP.
 * Holds its ID, name, domain of values, and current assignment.
 */
public class Variable {
    public int id;
    public String name;
    public List<Object> domain;
    public Object assignedValue = null;

    // For GBJ
    public Set<Integer> ancestors = new HashSet<>();

    public Variable(int id, String name, List<Object> domain) {
        this.id = id;
        this.name = name;
        this.domain = new ArrayList<>(domain);
    }

    /**
     * Returns the current size of the domain.
     * 
     * @return Domain size.
     */
    public int domainSize() {
        return domain.size();
    }
}
