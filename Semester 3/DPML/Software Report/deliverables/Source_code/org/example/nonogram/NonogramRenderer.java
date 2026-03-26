package org.example.nonogram;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.HashMap;
import javax.imageio.ImageIO;

/**
 * Render a solved Nonogram as a BufferedImage and optionally save as PNG.
 *
 * Usage:
 * BufferedImage img = NonogramRenderer.renderFromSolutionMap(solutionMap, rows,
 * cols, 20);
 * NonogramRenderer.saveImage(img, "out.png");
 */
public class NonogramRenderer {

    /**
     * Render the solution from a map of Variable Name -> Value.
     * Value should be int[] (or Integer[]).
     */
    public static BufferedImage renderFromSolutionMap(Map<String, Object> solution, int rows, int cols, int cellSize) {
        // Build a map from variable name -> int[] pattern
        Map<String, int[]> varMap = new HashMap<>();
        for (Map.Entry<String, Object> e : solution.entrySet()) {
            String name = e.getKey();
            Object val = e.getValue();
            if (val instanceof int[]) {
                varMap.put(name, (int[]) val);
            } else if (val instanceof Integer[]) {
                // unlikely, but handle boxed ints
                Integer[] boxed = (Integer[]) val;
                int[] arr = new int[boxed.length];
                for (int i = 0; i < boxed.length; i++)
                    arr[i] = boxed[i];
                varMap.put(name, arr);
            } else if (val instanceof String) {
                // Handle String representation "01010"
                String s = (String) val;
                int[] arr = new int[s.length()];
                for (int i = 0; i < s.length(); i++)
                    arr[i] = s.charAt(i) - '0';
                varMap.put(name, arr);
            }
        }

        // Build grid of ints (rows x cols). Try to read Row variables R0..R{rows-1}
        int[][] grid = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            String rname = "R" + i;
            int[] pattern = varMap.get(rname);
            if (pattern == null) {
                throw new IllegalArgumentException("Cannot find row variable '" + rname + "' in solution.");
            }
            if (pattern.length != cols) {
                throw new IllegalArgumentException("Row pattern length mismatch for " + rname
                        + ": expected " + cols + " got " + pattern.length);
            }
            System.arraycopy(pattern, 0, grid[i], 0, cols);
        }

        return renderGrid(grid, cellSize);
    }

    /**
     * Render an integer grid (0=white, 1=black) to BufferedImage.
     */
    public static BufferedImage renderGrid(int[][] grid, int cellSize) {
        int rows = grid.length;
        int cols = grid.length > 0 ? grid[0].length : 0;
        int w = cols * cellSize + 1;
        int h = rows * cellSize + 1;
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        try {
            // white background
            g.setColor(Color.WHITE);
            g.fillRect(0, 0, w, h);

            // fill black squares
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (grid[i][j] != 0) {
                        int x = j * cellSize;
                        int y = i * cellSize;
                        g.setColor(Color.BLACK);
                        g.fillRect(x + 1, y + 1, cellSize - 1, cellSize - 1);
                    }
                }
            }

            // draw grid lines
            g.setColor(Color.GRAY);
            for (int i = 0; i <= rows; i++) {
                int y = i * cellSize;
                g.drawLine(0, y, cols * cellSize, y);
            }
            for (int j = 0; j <= cols; j++) {
                int x = j * cellSize;
                g.drawLine(x, 0, x, rows * cellSize);
            }
        } finally {
            g.dispose();
        }
        return img;
    }

    /**
     * Save buffered image to PNG path.
     * 
     * @param img  BufferedImage
     * @param path output path (e.g., "out.png")
     * @throws IOException on write failure
     */
    public static void saveImage(BufferedImage img, String path) throws IOException {
        File out = new File(path);
        ImageIO.write(img, "PNG", out);
    }
}
