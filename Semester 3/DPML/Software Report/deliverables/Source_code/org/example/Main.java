package org.example;

import org.example.planner.DinnerPartyTestSuite;
import org.example.nonogram.NonogramTestSuite;

public class Main {
    public static void main(String[] args) {
        System.out.println("\n>>> Running Dinner Party Test Suite...");
        DinnerPartyTestSuite.run();

         System.out.println("\n>>> Running Nonogram Test Suite...");
         NonogramTestSuite.run();
    }
}
