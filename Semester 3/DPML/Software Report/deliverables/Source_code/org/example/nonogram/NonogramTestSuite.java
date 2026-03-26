package org.example.nonogram;

import org.example.csp.*;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.example.common.AbstractTestSuite;

/**
 * Test suite for the Nonogram Problem.
 * Loads puzzles, runs benchmarks, and saves results/images.
 */
public class NonogramTestSuite extends AbstractTestSuite<NonogramFileLoader.Puzzle, Map<String, Object>> {

    /**
     * Runs the Nonogram test suite.
     */
    public static void run() {
        new NonogramTestSuite().execute();
    }

    @Override
    protected String getInputDirPath() {
        return "src/main/resources/nonograms/input";
    }

    @Override
    protected String getOutputDirPath() {
        return "src/main/resources/nonograms/output";
    }

    @Override
    protected String getFileExtension() {
        return ".txt";
    }

    @Override
    protected NonogramFileLoader.Puzzle loadConfiguration(String resourcePath) throws IOException {
        return NonogramFileLoader.loadFromResource(resourcePath);
    }

    @Override
    protected CSP createCSP(NonogramFileLoader.Puzzle puzzle) {
        return NonogramProblem.create(puzzle.rows(), puzzle.cols(), puzzle.rowClues(), puzzle.colClues());
    }

    @Override
    protected Map<String, Object> captureSolution(CSP csp) {
        Map<String, Object> sol = new HashMap<>();
        for (Variable v : csp.variables) {
            sol.put(v.name, v.assignedValue);
        }
        return sol;
    }

    @Override
    protected String getDimensions(NonogramFileLoader.Puzzle config) {
        return config.rows() + "x" + config.cols();
    }

    @Override
    protected String getFileOrConfigId(NonogramFileLoader.Puzzle config) {
        return config.fileName();
    }

    @Override
    protected void renderOutput(Map<String, Object> solution, String outputDir, NonogramFileLoader.Puzzle config)
            throws IOException {
        BufferedImage img = NonogramRenderer.renderFromSolutionMap(solution,
                config.rows(), config.cols(), 20);
        String outName = config.fileName().replace(".txt", ".png");
        NonogramRenderer.saveImage(img, outputDir + "/" + outName);
    }
}
