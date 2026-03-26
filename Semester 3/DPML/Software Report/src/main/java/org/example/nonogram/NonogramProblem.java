package org.example.nonogram;

import org.example.csp.*;
import java.util.*;

/**
 * Models a Nonogram puzzle as a Constraint Satisfaction Problem.
 * Variables represent rows and columns.
 * Domains are the set of all valid patterns for a given row/column clue.
 * Constraints ensure that the row and column patterns match at their
 * intersection.
 */
public class NonogramProblem {

    /**
     * Helper to generate valid line patterns (simplified for small grids).
     * 
     * @param length Length of the line (row or column).
     * @param clue   The clues for this line.
     * @return List of valid patterns.
     */
    private static List<Object> generatePatterns(int length, int[] clue) {
        List<Object> valid = new ArrayList<>();
        recursePattern(length, clue, 0, new int[length], 0, valid);
        return valid;
    }

    private static void recursePattern(int len, int[] clue, int cIdx, int[] current, int pos, List<Object> valid) {
        if (cIdx == clue.length) {
            // Fill remaining with 0
            for (int i = pos; i < len; i++)
                current[i] = 0;
            // Convert to string for storage
            StringBuilder sb = new StringBuilder();
            for (int x : current)
                sb.append(x);
            valid.add(sb.toString());
            return;
        }

        // Try placing clue[cIdx] at start index 's'
        int run = clue[cIdx];
        // Remaining blocks + remaining spaces needed
        int remain = 0;
        for (int k = cIdx + 1; k < clue.length; k++)
            remain += clue[k] + 1;

        for (int s = pos; s <= len - remain - run; s++) {
            for (int k = pos; k < s; k++)
                current[k] = 0;
            for (int k = s; k < s + run; k++)
                current[k] = 1;
            if (s + run < len)
                current[s + run] = 0;

            recursePattern(len, clue, cIdx + 1, current, s + run + 1, valid);
        }
    }

    public static CSP create(int rows, int cols, List<List<Integer>> rowClues, List<List<Integer>> colClues) {
        CSP csp = new CSP();

        // Rows are variables 0 to rows-1
        for (int r = 0; r < rows; r++) {
            int[] clueArr = rowClues.get(r).stream().mapToInt(i -> i).toArray();
            List<Object> dom = generatePatterns(cols, clueArr);
            csp.addVariable(new Variable(r, "R" + r, dom));
        }

        // Cols are variables rows to rows+cols-1
        for (int c = 0; c < cols; c++) {
            int[] clueArr = colClues.get(c).stream().mapToInt(i -> i).toArray();
            List<Object> dom = generatePatterns(rows, clueArr);
            csp.addVariable(new Variable(rows + c, "C" + c, dom));
        }

        // Constraints: Intersection at (r, c) must match
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                final int colIdx = c; // char index in row string (0..cols-1)
                final int rowIdx = r; // char index in col string (0..rows-1)

                csp.addConstraint(new Constraint(r, rows + c) {
                    public boolean isSatisfied(Object valRow, Object valCol) {
                        String sRow = (String) valRow;
                        String sCol = (String) valCol;
                        return sRow.charAt(colIdx) == sCol.charAt(rowIdx);
                    }
                });
            }
        }
        return csp;
    }
}
