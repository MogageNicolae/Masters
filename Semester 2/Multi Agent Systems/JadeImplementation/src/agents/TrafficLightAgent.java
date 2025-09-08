package agents;

import jade.core.Agent;
import jade.lang.acl.ACLMessage;
import jade.core.behaviours.CyclicBehaviour;
import jade.core.behaviours.TickerBehaviour;

public class TrafficLightAgent extends Agent {

    private int greenTime = 0;
    private int redTime = 0;
    private String currentState = "RED";

    protected void setup() {
        System.out.println(getLocalName() + " started.");

        addBehaviour(new CyclicBehaviour() {
            public void action() {
                ACLMessage msg = receive();
                if (msg != null) {
                    String content = msg.getContent();
                    System.out.println(getLocalName() + " received command: " + content);

                    if (content.startsWith("SwitchGreen:")) {
                        greenTime = Integer.parseInt(content.split(":")[1]);
                        redTime = 0;
                        currentState = "GREEN";
                    } else if (content.startsWith("SwitchRed:")) {
                        redTime = Integer.parseInt(content.split(":")[1]);
                        greenTime = 0;
                        currentState = "RED";
                    } else if (content.startsWith("KeepGreen:")) {
                        greenTime = Integer.parseInt(content.split(":")[1]);
                        redTime = 0;
                        currentState = "GREEN";
                    } else if (content.startsWith("KeepRed:")) {
                        redTime = Integer.parseInt(content.split(":")[1]);
                        greenTime = 0;
                        currentState = "RED";
                    }
                } else {
                    block();
                }
            }
        });

        addBehaviour(new TickerBehaviour(this, 2000) {
            protected void onTick() {
                System.out.println(getLocalName() + " current state: " + currentState + " | GreenTime left: " + greenTime + " | RedTime left: " + redTime);
                if (greenTime > 0) greenTime -= 2;
                if (redTime > 0) redTime -= 2;
                if (greenTime < 0) greenTime = 0;
                if (redTime < 0) redTime = 0;
            }
        });
    }
}
