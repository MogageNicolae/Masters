package org.example.planner;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * Loads Dinner Party configurations from resource files.
 *
 * File format (Java properties):
 * - no_guests=<number>
 * - no_tables=<number>
 * - capacity=<number>
 * - enemyProb=<decimal>
 * - partnerProb=<decimal>
 * - politicsProb=<decimal>
 * - configId=<identifier>
 */
public class DinnerPartyFileLoader {

    public record Configuration(
            int noGuests,
            int noTables,
            int capacity,
            double enemyProb,
            double partnerProb,
            double politicsProb,
            String configId) {
    }

    /**
     * Load a configuration from a file path.
     */
    public static Configuration loadFromResource(String filePath) throws IOException {
        Properties props = new Properties();

        try (InputStream is = new java.io.FileInputStream(filePath)) {
            props.load(is);
        }

        // Parse properties into configuration record
        int noGuests = Integer.parseInt(props.getProperty("no_guests"));
        int noTables = Integer.parseInt(props.getProperty("no_tables"));
        int capacity = Integer.parseInt(props.getProperty("capacity"));
        double enemyProb = Double.parseDouble(props.getProperty("enemyProb"));
        double partnerProb = Double.parseDouble(props.getProperty("partnerProb"));
        double politicsProb = Double.parseDouble(props.getProperty("politicsProb"));
        String configId = props.getProperty("configId");

        return new Configuration(noGuests, noTables, capacity, enemyProb, partnerProb, politicsProb, configId);
    }
}
