package agents;

import jade.core.Agent;
import jade.core.AID;
import jade.lang.acl.ACLMessage;
import jade.core.behaviours.TickerBehaviour;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;


public class TrafficSensorAgent extends Agent {
    private String filepath;
    private int currentTickStep = 0;
    private List<int[]> trafficStages = new ArrayList<>();

    protected void setup() {
        Object[] args = getArguments();
        if (args != null && args.length > 0) {
            filepath = (String) args[0];
            loadTrafficMatrix(filepath);
        } else {
            System.out.println("No arguments given");
            doDelete();
        }
        System.out.println(getLocalName() + " started. Reading from " + filepath);

        addBehaviour(new TickerBehaviour(this, 2000) {
            @Override
            protected void onTick() {
                if (trafficStages.isEmpty()) {
                    System.out.println(getLocalName()+ ": No traffic stage found");
                    stop();
                    doDelete();
                }

                if (currentTickStep < trafficStages.getFirst().length) {
                    int vehicleCount = 0;
                    for (int[] row : trafficStages) {
                        vehicleCount += row[currentTickStep];
                    }

                    ACLMessage msg = new ACLMessage(ACLMessage.INFORM);
                    msg.addReceiver(new AID("TrafficControlAgent", AID.ISLOCALNAME));
                    msg.setContent(getLocalName() + ":VehicleCount:" + vehicleCount);
                    send(msg);
                    System.out.println(getLocalName() + " sent count: " + vehicleCount);
                    currentTickStep++;
                }
                else {
                    System.out.println(getLocalName() + ": Finished sending all stages.");
                    stop();
                    doDelete();
                }
            }
        });

    }
    private void loadTrafficMatrix(String filePath) {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] tokens = line.trim().split("\\s+");
                int[] row = new int[tokens.length];
                for (int i = 0; i < tokens.length; i++) {
                    row[i] = Integer.parseInt(tokens[i]);
                }
                trafficStages.add(row);
            }
        } catch (Exception e) {
            System.out.println(getLocalName() + ": Error reading traffic file: " + e.getMessage());
        }
    }
}
