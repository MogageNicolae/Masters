package firesystem.firesensor;

import agent.Action;
import agent.Agent;
import agent.Environment;
import agent.Percept;
import firesystem.actions.SendMessage;
import firesystem.communication.AgentID;
import firesystem.communication.Message;
import firesystem.communication.MessageType;
import firesystem.communication.payloads.FireAlertPayload;
import firesystem.communication.payloads.FireExtinguishedPayload;
import firesystem.utils.Position;

import java.util.List;
import java.util.ArrayList;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

public class FireSensorAgent extends Agent implements Runnable {
    private final AgentID agentID;
    private final List<Position> reportedFirePositions;
    private final Environment environment;
    private final Queue<Message> inbox;
    private final Queue<Action> outgoingActions;

    private volatile boolean running = true;

    public FireSensorAgent(AgentID agentID, Environment env) {
        this.agentID = agentID;
        this.environment = env;
        this.inbox = new ConcurrentLinkedQueue<>();
        this.outgoingActions = new ConcurrentLinkedQueue<>();
        this.reportedFirePositions = new ArrayList<>();
    }

    public Integer getAgentID() {
        return agentID.getId();
    }

    @Override
    public void receiveMessage(Message msg) {
        inbox.offer(msg);
    }

    @Override
    public void think(Percept p) {
        if (p instanceof FireSensorPercept sp) {
            List<Position> currentFirePositions = sp.getFirePositions();

            for (Position position : currentFirePositions) {
                if (!reportedFirePositions.contains(position)) {
                    System.out.println(agentID + " found new fire at (" + position.getX() + "," + position.getY() + ")");
                    reportedFirePositions.add(position);

                    FireAlertPayload payload = new FireAlertPayload(position);
                    Message fireAlertMessage = new Message(agentID, new AgentID(0, "FCA"), MessageType.FIRE_ALERT, payload);
                    outgoingActions.offer(new SendMessage(fireAlertMessage, environment));
                }
            }
        }
    }

    @Override
    public Action selectAction() {
        processInbox();

        if (!outgoingActions.isEmpty()) {
            return outgoingActions.poll();
        }

        return null;
    }

    private void processInbox() {
        while (!inbox.isEmpty()) {
            Message msg = inbox.poll();
//            System.out.println(agentID + " processing message: " + msg.getType() + " from " + msg.getSender());

            if (msg.getType() == MessageType.FIRE_EXTINGUISHED) {
                if (msg.getPayload() instanceof FireExtinguishedPayload payload) {
                    Position extinguishedSpot = payload.getExtinguishedPosition();
                    if (reportedFirePositions.contains(extinguishedSpot)) {
                        reportedFirePositions.remove(extinguishedSpot);
                        System.out.println(agentID + ": Confirmed fire extinguished at " + extinguishedSpot + ". Looking for more...");
                    }
                }
            }
        }
    }

    public void stopRunning() {
        running = false;
    }

    @Override
    public void run() {
        System.out.println(agentID + " started running.");
        while (running) {
            try {
                think(environment.getPercept(this));

                Action action = selectAction();

                if (action != null) {
                    environment.updateState(this, action);
                }

                Thread.sleep(2000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                System.out.println(agentID + " interrupted.");
                running = false;
            } catch (Exception e) {
                System.err.println(agentID + " encountered an error: " + e.getMessage());
                e.printStackTrace();
                running = false;
            }
        }
        System.out.println(agentID + " stopped running.");
    }
}
