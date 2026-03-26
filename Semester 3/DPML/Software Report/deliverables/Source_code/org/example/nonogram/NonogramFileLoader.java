package org.example.nonogram;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Loads Nonogram puzzles from resource files.
 *
 * File format:
 * - First line: No_of_Rows No_of_Columns
 * - Next R lines: row clues (space-separated integers;)
 * - Next C lines: column clues (space-separated integers;)
 */
public class NonogramFileLoader {

    public record Puzzle(int rows, int cols, List<List<Integer>> rowClues, List<List<Integer>> colClues,
            String fileName) {
    }

    /**
     * Load a puzzle from a file path.
     */
    public static Puzzle loadFromResource(String filePath) throws IOException {
        // Extract filename from file path
        String fileName = filePath.substring(filePath.lastIndexOf('/') + 1);

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new java.io.FileInputStream(filePath), StandardCharsets.UTF_8))) {
            List<String> lines = reader.lines().toList();
            List<String> tokens = new ArrayList<>();
            for (String line : lines) {
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#"))
                    continue;
                tokens.add(trimmed);
            }
            if (tokens.isEmpty()) {
                throw new IOException("Empty puzzle file: " + filePath);
            }

            // First line: R C
            String[] rc = tokens.getFirst().split("\\s+");
            if (rc.length != 2)
                throw new IOException("First line must be 'R C': " + tokens.getFirst());
            int R = Integer.parseInt(rc[0]);
            int C = Integer.parseInt(rc[1]);

            if (tokens.size() < 1 + R + C) {
                throw new IOException("Not enough lines for clues: expected " + (1 + R + C) + ", got " + tokens.size());
            }

            List<List<Integer>> rowClues = new ArrayList<>();
            for (int i = 0; i < R; i++) {
                rowClues.add(parseClueLine(tokens.get(1 + i)));
            }
            List<List<Integer>> colClues = new ArrayList<>();
            for (int j = 0; j < C; j++) {
                colClues.add(parseClueLine(tokens.get(1 + R + j)));
            }

            return new Puzzle(R, C, rowClues, colClues, fileName);
        }
    }

    private static List<Integer> parseClueLine(String line) throws IOException {
        try {
            List<Integer> clues = Arrays.stream(line.split("\\s+"))
                    .map(Integer::parseInt)
                    .collect(Collectors.toList());
            if (clues.size() == 1 && clues.getFirst() == 0) {
                return List.of(0);
            }
            return clues;
        } catch (NumberFormatException e) {
            throw new IOException("Invalid clue line: '" + line + "'", e);
        }
    }
}
