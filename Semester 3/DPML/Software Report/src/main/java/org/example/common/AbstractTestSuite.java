package org.example.common;

import org.example.csp.*;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Abstract base class for Test Suites.
 * Encapsulates common benchmarking logic.
 *
 * @param <T> The type of the configuration object (e.g., Puzzle, Map).
 * @param <S> The type of the solution object (e.g., Map, CSP).
 */
public abstract class AbstractTestSuite<T, S> {

    /**
     * Creates a CSP instance from the configuration object.
     */
    protected abstract CSP createCSP(T config);

    /**
     * Captures the solution from the solved CSP.
     */
    protected abstract S captureSolution(CSP csp);

    /**
     * Returns the dimensions string for the benchmark report.
     * Example: "5x5" for Nonogram, "100x10x15" for Dinner Party
     */
    protected abstract String getDimensions(T config);

    /**
     * Returns the file or configuration identifier.
     * Example: "puzzle01.txt" for Nonogram, "cfg_001" for Dinner Party
     */
    protected abstract String getFileOrConfigId(T config);

    /**
     * Returns the input directory path for loading configurations.
     * Can be a resource path or an absolute/relative file system path.
     */
    protected abstract String getInputDirPath();

    /**
     * Returns the output directory path for saving results.
     */
    protected abstract String getOutputDirPath();

    /**
     * Returns the file extension filter for input files.
     */
    protected abstract String getFileExtension();

    /**
     * Loads a configuration from the given resource path.
     */
    protected abstract T loadConfiguration(String resourcePath) throws IOException;

    /**
     * Renders and saves the output (solution visualization or report).
     */
    protected abstract void renderOutput(S solution, String outputDir, T config) throws IOException;

    /**
     * Main execution method for running the test suite.
     */
    public void execute() {
        System.out.println("Starting Test Suite...");

        String inputDirPath = getInputDirPath();
        String outputDir = getOutputDirPath();
        String reportFileName = "benchmark_results.md";
        String reportPath = outputDir + "/" + reportFileName;

        String header = "#Benchmark Results" + System.lineSeparator() + System.lineSeparator() +
                "| File | Dimensions | Vars | Constraints | Density | Method | Solved | Reduction Time | Search Time | Total Time | Nodes | Checks | Backtracks |"
                + System.lineSeparator() +
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|" + System.lineSeparator();

        File[] files = BenchmarkRunner.setupEnvironment(inputDirPath, outputDir, reportFileName,
                header, getFileExtension());

        if (files.length == 0)
            return;

        for (File file : files) {
            System.out.println("Processing: " + file.getName());
            try {
                T config = loadConfiguration(inputDirPath + "/" + file.getName());
                runBenchmark(config, reportPath, outputDir);
            } catch (OutOfMemoryError e) {
                System.err.println("FATAL: OutOfMemoryError while processing " + file.getName());
                System.err.println("  Skipping this file and continuing with next...");
                System.err.println("  Tip: Try increasing heap size with -Xmx flag (e.g., java -Xmx4G ...)");
                // Force garbage collection to try to recover
                System.gc();
                try {
                    Thread.sleep(1000); // Give GC time to work
                } catch (InterruptedException ignored) {
                }
            } catch (Exception e) {
                System.err.println("Error processing " + file.getName());
                e.printStackTrace();
            }
        }
        System.out.println("\nBenchmark report saved to: " + reportPath);
    }

    /**
     * Runs benchmark for a single problem configuration.
     */
    protected void runBenchmark(T config, String reportPath, String outputDir) throws IOException {
        // Calculate metrics
        CSP cspBase = createCSP(config);
        int vars = cspBase.variables.size();
        int constraints = cspBase.constraints.size();
        double density = (vars > 1) ? (2.0 * constraints) / (vars * (vars - 1)) : 0;

        String dimensions = getDimensions(config);
        String fileOrConfigId = getFileOrConfigId(config);
        String info = String.format("| %s | %s | %d | %d | %.4f |",
                fileOrConfigId, dimensions, vars, constraints, density);

        // Shared reference to capture first successful solution
        AtomicReference<S> foundSolution = new AtomicReference<>();

        // Define strategies to run
        List<Callable<String>> tasks = new ArrayList<>();

        // 1. Baseline (No Reduction)
        addBaselineBenchmarks(info, config, tasks, foundSolution);

        // 2. AC3 Phase
        addReductionBenchmarks(info, config, "AC3", new Reductions.AC3(), tasks, foundSolution);

        // 3. PC4 Phase
        addReductionBenchmarks(info, config, "PC4", new Reductions.PC4(), tasks, foundSolution);

        // Execute all benchmarks in parallel
        BenchmarkRunner.executeParallelBenchmarks(tasks, reportPath);

        // Save output if solution found
        if (foundSolution.get() != null) {
            renderOutput(foundSolution.get(), outputDir, config);
        }
    }

    /**
     * Adds baseline benchmarks without reduction techniques.
     */
    protected void addBaselineBenchmarks(
            String info,
            T config,
            List<Callable<String>> tasks,
            AtomicReference<S> foundSolution) {

        tasks.add(() -> runSingleBenchmark(info, "Forward Checking", config,
                c -> new CSPSolver(c).setSearch(new SearchMethods.ForwardChecking()),
                0, null, null, foundSolution));
        tasks.add(() -> runSingleBenchmark(info, "Backtracking", config,
                c -> new CSPSolver(c).setSearch(new SearchMethods.Backtracking()),
                0, null, null, foundSolution));
        tasks.add(() -> runSingleBenchmark(info, "GBJ", config,
                c -> new CSPSolver(c).setSearch(new SearchMethods.GBJ()),
                0, null, null, foundSolution));
        tasks.add(() -> runSingleBenchmark(info, "GENET", config,
                c -> new CSPSolver(c).setSearch(new SearchMethods.GENET(10_000)),
                0, null, null, foundSolution));
    }

    /**
     * Adds benchmarks for a specific reduction technique (e.g., AC3, PC4).
     */
    protected void addReductionBenchmarks(
            String info,
            T config,
            String reductionName,
            Reduction reduction,
            List<Callable<String>> tasks,
            AtomicReference<S> foundSolution) {

        System.out.println("  Running " + reductionName + " Pre-computation...");

        ExecutorService executor = Executors.newSingleThreadExecutor();

        long start = System.currentTimeMillis();
        try {
            Future<CSPSolver> solverFuture = executor.submit(() -> {
                CSPSolver solver = new CSPSolver(createCSP(config)).addReduction(reduction);
                solver.solve();
                return solver;
            });

            CSPSolver solver = solverFuture.get(10, TimeUnit.MINUTES);
            long time = System.currentTimeMillis() - start;
            System.out.println("  " + reductionName + " Pre-computation finished in " + time + "ms");

            boolean possible = true;
            for (Variable v : solver.csp.variables) {
                if (v.domain.isEmpty()) {
                    possible = false;
                    break;
                }
            }

            if (possible) {
                Map<Integer, List<Object>> domains = new HashMap<>();
                for (Variable v : solver.csp.variables) {
                    domains.put(v.id, new ArrayList<>(v.domain));
                }
                Map<Constraint, Set<String>> prunedPairs = solver.getPrunedPairs();

                String prefix = reductionName + " + ";
                tasks.add(() -> runSingleBenchmark(info, prefix + "BT", config,
                        c -> new CSPSolver(c).setSearch(new SearchMethods.Backtracking()),
                        time, domains, prunedPairs, foundSolution));
                tasks.add(() -> runSingleBenchmark(info, prefix + "GBJ", config,
                        c -> new CSPSolver(c).setSearch(new SearchMethods.GBJ()),
                        time, domains, prunedPairs, foundSolution));
                tasks.add(() -> runSingleBenchmark(info, prefix + "FC", config,
                        c -> new CSPSolver(c).setSearch(new SearchMethods.ForwardChecking()),
                        time, domains, prunedPairs, foundSolution));
                tasks.add(() -> runSingleBenchmark(info, prefix + "GENET", config,
                        c -> new CSPSolver(c).setSearch(new SearchMethods.GENET(10_000)), // Default GENET limit
                        time, domains, prunedPairs, foundSolution));
            } else {
                System.out.println("  " + reductionName + " found inconsistency. Skipping benchmarks.");
            }

        } catch (TimeoutException e) {
            System.err.println("  " + reductionName + " Pre-computation timed out after 10 minutes. Skipping.");
            executor.shutdownNow();
        } catch (ExecutionException e) {
            System.err.println("  " + reductionName + " Pre-computation failed: " + e.getMessage());
            if (e.getCause() instanceof OutOfMemoryError) {
                System.err.println("  (OutOfMemoryError detected - problem too large for available memory)");
                System.err.println("  Tip: Try increasing heap size with -Xmx flag (e.g., java -Xmx4G ...)");
            }
            executor.shutdownNow();
        } catch (OutOfMemoryError e) {
            System.err.println("  " + reductionName + " Pre-computation failed: OutOfMemoryError");
            System.err.println("  (Problem too large for available memory)");
            System.err.println("  Tip: Try increasing heap size with -Xmx flag (e.g., java -Xmx4G ...)");
            executor.shutdownNow();
            System.gc();
        } catch (Exception e) {
            System.err.println("  " + reductionName + " Pre-computation failed: " + e.getMessage());
            e.printStackTrace();
            executor.shutdownNow();
        } finally {
            executor.shutdown();
        }
    }

    /**
     * Runs a single benchmark with the given parameters.
     */
    protected String runSingleBenchmark(
            String info,
            String methodName,
            T config,
            SolverFactory factory,
            long reductionTime,
            Map<Integer, List<Object>> reducedDomains,
            Map<Constraint, Set<String>> prunedPairs,
            AtomicReference<S> foundSolution) {

        try {
            CSP csp = createCSP(config);

            if (reducedDomains != null) {
                for (Variable v : csp.variables) {
                    if (reducedDomains.containsKey(v.id)) {
                        v.domain = new ArrayList<>(reducedDomains.get(v.id));
                    }
                }
            }

            CSPSolver solver = factory.create(csp);

            if (prunedPairs != null) {
                Map<Constraint, Set<String>> mappedPrunedPairs = new HashMap<>();
                for (Map.Entry<Constraint, Set<String>> entry : prunedPairs.entrySet()) {
                    Constraint oldC = entry.getKey();
                    Constraint newC = csp.getConstraint(oldC.var1Idx, oldC.var2Idx);
                    if (newC != null) {
                        mappedPrunedPairs.put(newC, entry.getValue());
                    }
                }
                solver.setPrunedPairs(mappedPrunedPairs);
            }

            return BenchmarkRunner.runWithTimeout(
                    info,
                    methodName,
                    solver,
                    reductionTime,
                    () -> {
                        if (foundSolution.get() == null) {
                            foundSolution.compareAndSet(null, captureSolution(csp));
                        }
                    });

        } catch (OutOfMemoryError e) {
            System.err.println("  [" + methodName + "] OutOfMemoryError during benchmark setup/execution");
            // Force garbage collection
            System.gc();
            // Return a result indicating failure
            return info + String.format(" %s | %s | %d | %d | %d | %d | %d | %d |" + System.lineSeparator(),
                    methodName, "OOM", reductionTime, 0, reductionTime, 0, 0, 0);
        } catch (Exception e) {
            System.err.println("  [" + methodName + "] Unexpected error: " + e.getMessage());
            return info + String.format(" %s | %s | %d | %d | %d | %d | %d | %d |" + System.lineSeparator(),
                    methodName, "Error", reductionTime, 0, reductionTime, 0, 0, 0);
        }
    }
}
