package agents;

import jade.core.Agent;
import jade.core.AID;
import jade.core.behaviours.TickerBehaviour;
import jade.lang.acl.ACLMessage;
import jade.core.behaviours.CyclicBehaviour;
import java.util.HashMap;
import java.util.Map;

public class TrafficControlAgent extends Agent {

    // vehicle counts from sensors
    private Map<String, Integer> sensorData = new HashMap<>();

    // states for two TLAs
    private Map<String, String> lightStates = new HashMap<>();
    private Map<String, Integer> remainingTimes = new HashMap<>();

    private final int MIN_GREEN_TIME = 5;
    private final int MAX_GREEN_TIME = 30;

    protected void setup() {
        System.out.println(getLocalName() + " started.");

        lightStates.put("TrafficLightAgent1", "GREEN");
        lightStates.put("TrafficLightAgent2", "RED");
        remainingTimes.put("TrafficLightAgent1", MAX_GREEN_TIME);
        remainingTimes.put("TrafficLightAgent2", MAX_GREEN_TIME);

        addBehaviour(new CyclicBehaviour() {
            @Override
            public void action() {
                ACLMessage msg = receive();
                if (msg != null) {
                    String content = msg.getContent();
                    System.out.println(getLocalName() + " received: " + content);

                    String[] parts = content.split(":");
                    if (parts.length == 3 && parts[1].equals("VehicleCount")) {
                        String sensorName = parts[0];
                        int count = Integer.parseInt(parts[2]);
                        sensorData.put(sensorName, count);
                    }

                    if (sensorData.size() == 2) {
                        adjustTimings();
                        sensorData.clear();
                    }
                }
                else {
                    block();
                }
            }
        });

        addBehaviour(new TickerBehaviour(this, 2000) {
            @Override
            protected void onTick() {
                for (String tla : remainingTimes.keySet()) {
                    int timeLeft = remainingTimes.get(tla);
                    if (timeLeft > 0) {
                        remainingTimes.put(tla, timeLeft - 2);
                    }
                }
            }
        });

    }
    private void adjustTimings() {
        int count1 = sensorData.getOrDefault("TrafficSensorAgent1", 0);
        int count2 = sensorData.getOrDefault("TrafficSensorAgent2", 0);

        String tla1 = "TrafficLightAgent1";
        String tla2 = "TrafficLightAgent2";

        String state1 = lightStates.get(tla1);
        String state2 = lightStates.get(tla2);

        int remTime1 = remainingTimes.get(tla1);
        int remTime2 = remainingTimes.get(tla2);

        System.out.println(getLocalName() + " adjusting timings with counts: " + count1 + " and " + count2);
        ACLMessage toTLA1 = new ACLMessage(ACLMessage.INFORM);
        toTLA1.addReceiver(new AID(tla1, AID.ISLOCALNAME));
        ACLMessage toTLA2 = new ACLMessage(ACLMessage.INFORM);
        toTLA2.addReceiver(new AID(tla2, AID.ISLOCALNAME));

        //calc total traffic and fraction per side
        int totalCars = count1 + count2;
        if (totalCars == 0) totalCars = 1;
        double frac1 = (double) count1 / totalCars;
        double frac2 = (double) count2 / totalCars;

        //calc suggested green times proportional to cars, capped min/max
        int suggestedGreen1 = (int) Math.max(MIN_GREEN_TIME, Math.min(MAX_GREEN_TIME, frac1 * MAX_GREEN_TIME));
        int suggestedGreen2 = (int) Math.max(MIN_GREEN_TIME, Math.min(MAX_GREEN_TIME, frac2 * MAX_GREEN_TIME));

        //if other side now has much higher traffic AND current green remaining is > MIN_GREEN_TIME, switch early
        if (state1.equals("GREEN") && count2 > count1 * 2 && remTime1 > MIN_GREEN_TIME) {
            System.out.println(getLocalName() + ": Preempting TLA1 green for TLA2 due to traffic shift");
            //switch : TLA1 -> red, TLA2 -> green
            toTLA1.setContent("SwitchRed:" + suggestedGreen2);
            toTLA2.setContent("SwitchGreen:" + suggestedGreen2);

            lightStates.put(tla1, "RED");
            lightStates.put(tla2, "GREEN");

            remainingTimes.put(tla1, suggestedGreen2);
            remainingTimes.put(tla2, suggestedGreen2);
        }
        else if (state2.equals("GREEN") && count1 > count2 * 2 && remTime2 > MIN_GREEN_TIME) {
            System.out.println(getLocalName() + ": Preempting TLA2 green for TLA1 due to traffic shift");
            toTLA2.setContent("SwitchRed:" + suggestedGreen1);
            toTLA1.setContent("SwitchGreen:" + suggestedGreen1);

            lightStates.put(tla2, "RED");
            lightStates.put(tla1, "GREEN");

            remainingTimes.put(tla2, suggestedGreen1);
            remainingTimes.put(tla1, suggestedGreen1);
        }
        else {
            // update timings to suggested values if current green is near ending
            if (state1.equals("GREEN") && remTime1 <= MIN_GREEN_TIME) {
                toTLA1.setContent("SwitchGreen:" + suggestedGreen1);
                toTLA2.setContent("SwitchRed:" + suggestedGreen2);

                lightStates.put(tla1, "GREEN");
                lightStates.put(tla2, "RED");

                remainingTimes.put(tla1, suggestedGreen1);
                remainingTimes.put(tla2, suggestedGreen1);
            }
            else if (state2.equals("GREEN") && remTime2 <= MIN_GREEN_TIME) {
                toTLA2.setContent("SwitchGreen:" + suggestedGreen2);
                toTLA1.setContent("SwitchRed:" + suggestedGreen1);

                lightStates.put(tla2, "GREEN");
                lightStates.put(tla1, "RED");

                remainingTimes.put(tla2, suggestedGreen2);
                remainingTimes.put(tla1, suggestedGreen2);
            }
            else {
                if (state1.equals("GREEN")) {
                    toTLA1.setContent("KeepGreen:" + remTime1);
                    toTLA2.setContent("KeepRed:" + remTime2);
                } else {
                    toTLA2.setContent("KeepGreen:" + remTime2);
                    toTLA1.setContent("KeepRed:" + remTime1);
                }
            }
        }
        send(toTLA1);
        send(toTLA2);
    }
}
