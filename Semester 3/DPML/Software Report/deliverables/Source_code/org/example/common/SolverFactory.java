package org.example.common;

import org.example.csp.CSP;
import org.example.csp.CSPSolver;

/**
 * Factory interface for creating CSPSolver instances.
 */
public interface SolverFactory {
    CSPSolver create(CSP csp);
}
