package org.example.common;

import org.example.csp.CSPSolver;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.concurrent.*;

/**
 * Utility class for running CSP benchmarks.
 * Handles parallel execution, timeouts, and result reporting.
 */
public class BenchmarkRunner {

    private static final long TIMEOUT_MINUTES = 15;
    private static final long TIMEOUT_MS = TIMEOUT_MINUTES * 60 * 1000;

    /**
     * Executes a list of benchmark tasks in parallel and appends results to the
     * report file.
     *
     * @param tasks      List of callable tasks returning result strings.
     * @param reportPath Path to the markdown report file.
     */
    public static void executeParallelBenchmarks(List<Callable<String>> tasks, String reportPath) {
        ExecutorService suiteExecutor = Executors.newCachedThreadPool();
        ExecutorCompletionService<String> completionService = new ExecutorCompletionService<>(suiteExecutor);

        System.out.println(">>> Executing " + tasks.size() + " benchmarks in parallel...");
        for (Callable<String> task : tasks) {
            completionService.submit(task);
        }

        int completed = 0;
        int failed = 0;

        try {
            for (int i = 0; i < tasks.size(); i++) {
                try {
                    Future<String> resultFuture = completionService.take();
                    String result = resultFuture.get();

                    if (result != null) {
                        try {
                            Files.writeString(Path.of(reportPath), result, StandardOpenOption.APPEND);
                        } catch (IOException e) {
                            System.err.println("  Failed to write result to file: " + e.getMessage());
                        }
                    }

                    completed++;
                    System.out.println(">>> Completed " + completed + " / " + tasks.size() + " benchmarks");

                } catch (ExecutionException e) {
                    failed++;
                    Throwable cause = e.getCause();

                    if (cause instanceof OutOfMemoryError) {
                        System.err.println(">>> Benchmark " + (i + 1) + " failed: OutOfMemoryError");
                        System.err.println("    (Continuing with remaining benchmarks...)");
                        // Force garbage collection to try to recover
                        System.gc();
                    } else {
                        System.err.println(">>> Benchmark " + (i + 1) + " failed: " + cause.getClass().getSimpleName());
                        System.err.println("    Message: " + cause.getMessage());
                    }
                } catch (InterruptedException e) {
                    System.err.println(">>> Benchmark execution interrupted");
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        } finally {
            suiteExecutor.shutdownNow();
            try {
                if (!suiteExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                    System.err.println(">>> Some benchmark tasks did not terminate cleanly");
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        if (failed > 0) {
            System.out.println(">>> Benchmark summary: " + completed + " completed, " + failed + " failed");
        }
    }

    /**
     * Runs a single benchmark with a timeout.
     *
     * @param infoPrefix    String prefix for the result row (e.g. "| Config | ...
     *                      |").
     * @param methodName    Name of the algorithm being tested.
     * @param solver        The configured CSPSolver instance.
     * @param reductionTime Time taken for pre-reduction (AC3/PC4) in ms.
     * @param onSuccess     Callback to execute if solved successfully (e.g. to
     *                      capture
     *                      solution).
     * @return Formatted markdown table row with results.
     */
    public static String runWithTimeout(String infoPrefix, String methodName, CSPSolver solver, long reductionTime,
            Runnable onSuccess) {
        ExecutorService executor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r);
            t.setDaemon(true);
            return t;
        });
        Future<Boolean> future = executor.submit(solver::solve);

        long start = System.currentTimeMillis();
        boolean solved = false;
        String status = "FAILED";
        long searchDuration = 0;

        try {
            solved = future.get(TIMEOUT_MINUTES, TimeUnit.MINUTES);
            searchDuration = System.currentTimeMillis() - start;
            status = solved ? "SOLVED" : "FAILED";

            if (solved && onSuccess != null) {
                onSuccess.run();
            }
        } catch (TimeoutException e) {
            status = "TIMEOUT";
            future.cancel(true);
            searchDuration = TIMEOUT_MS;
        } catch (Exception e) {
            status = "ERROR";
            searchDuration = System.currentTimeMillis() - start;
        } finally {
            executor.shutdownNow();
        }

        long totalTime = reductionTime + searchDuration;

        System.out.printf(
                "  [%s] %s | Search Time: %dms | Total: %dms | Nodes: %d | Checks: %d | Backtracks: %d%n",
                methodName, status, searchDuration, totalTime, solver.nodesVisited,
                solver.constraintChecks, solver.backtrackCount);

        return infoPrefix + String.format(" %s | %s | %d | %d | %d | %d | %d | %d |" + System.lineSeparator(),
                methodName, status.equals("SOLVED") ? "Yes" : (status.equals("TIMEOUT") ? "Timeout" : "No"),
                reductionTime, searchDuration, totalTime, solver.nodesVisited, solver.constraintChecks,
                solver.backtrackCount);
    }

    /**
     * Sets up the benchmark environment: ensures output directory exists,
     * initializes report file, and discovers input files.
     *
     * @param inputResourcePath Path to input resources (e.g., "planner/input").
     * @param outputDirPath     Path to output directory (e.g.,
     *                          "src/main/resources/planner/output").
     * @param reportFileName    Name of the report file (e.g.,
     *                          "benchmark_results.md").
     * @param header            Markdown header for the report.
     * @param fileExtension     File extension to filter input files (e.g.,
     *                          ".config").
     * @return Array of discovered input files, sorted by name.
     */
    public static java.io.File[] setupEnvironment(String inputResourcePath, String outputDirPath, String reportFileName,
            String header, String fileExtension) {
        // Ensure output directory exists
        java.io.File outputDir = new java.io.File(outputDirPath);
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }

        String reportPath = outputDirPath + "/" + reportFileName;

        // Initialize report file
        try {
            Files.writeString(Path.of(reportPath), header);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // List files from resource directory
        java.io.File inputDir = new java.io.File(inputResourcePath);

        if (!inputDir.exists()) {
            ClassLoader cl = Thread.currentThread().getContextClassLoader();
            java.net.URL url = cl.getResource(inputResourcePath);
            if (url != null) {
                try {
                    inputDir = new java.io.File(url.toURI());
                } catch (Exception e) {
                    // ignore
                }
            }
        }

        if (!inputDir.exists()) {
            System.err.println("Could not find input directory: " + inputDir.getAbsolutePath());
            return new java.io.File[0];
        }

        java.io.File[] files = inputDir.listFiles((dir, name) -> name.endsWith(fileExtension));

        if (files == null || files.length == 0) {
            System.out.println("No input files found in " + inputDir.getAbsolutePath());
            return new java.io.File[0];
        }

        // Sort files by name for consistent ordering
        java.util.Arrays.sort(files, (a, b) -> a.getName().compareTo(b.getName()));

        return files;
    }
}
