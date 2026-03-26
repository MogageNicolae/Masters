package org.example.planner;

import org.example.common.AbstractTestSuite;
import org.example.csp.CSP;

import java.io.IOException;

/**
 * Test suite for the Dinner Party Problem.
 * Loads configurations, runs benchmarks, and saves results.
 */
public class DinnerPartyTestSuite extends AbstractTestSuite<DinnerPartyFileLoader.Configuration, CSP> {
    /**
     * Runs the Dinner Party test suite.
     */
    public static void run() {
        new DinnerPartyTestSuite().execute();
    }

    @Override
    protected String getInputDirPath() {
        return "src/main/resources/planner/input";
    }

    @Override
    protected String getOutputDirPath() {
        return "src/main/resources/planner/output";
    }

    @Override
    protected String getFileExtension() {
        return ".config";
    }

    @Override
    protected DinnerPartyFileLoader.Configuration loadConfiguration(String resourcePath) throws IOException {
        return DinnerPartyFileLoader.loadFromResource(resourcePath);
    }

    @Override
    protected CSP createCSP(DinnerPartyFileLoader.Configuration config) {
        return DinnerPartyProblem.create(config.noGuests(), config.noTables(), config.capacity(),
                config.enemyProb(), config.partnerProb(), config.politicsProb());
    }

    @Override
    protected CSP captureSolution(CSP csp) {
        return csp;
    }

    @Override
    protected String getDimensions(DinnerPartyFileLoader.Configuration config) {
        return String.format("%dx%dx%d", config.noGuests(), config.noTables(), config.capacity());
    }

    @Override
    protected String getFileOrConfigId(DinnerPartyFileLoader.Configuration config) {
        return config.configId();
    }

    @Override
    protected void renderOutput(CSP solution, String outputDir, DinnerPartyFileLoader.Configuration config)
            throws IOException {
        DinnerPartyRenderer.saveConstraintDistribution(solution, outputDir, config.configId());
    }
}
