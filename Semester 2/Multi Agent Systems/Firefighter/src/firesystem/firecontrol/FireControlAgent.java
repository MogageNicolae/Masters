package firesystem.firecontrol;

import agent.Action;
import agent.Agent;
import agent.Environment;
import agent.Percept;
import firesystem.actions.SendMessage;
import firesystem.communication.AgentID;
import firesystem.communication.Message;
import firesystem.communication.MessageType;
import firesystem.communication.payloads.DeployFirefighterPayload;
import firesystem.communication.payloads.FireAlertPayload;
import firesystem.communication.payloads.FireExtinguishedPayload;
import firesystem.firefighter.FirefighterAgent;
import firesystem.utils.Position;

import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

class FireReport {
    private final Position location;
    private final AgentID reporter;

    public FireReport(Position location, AgentID reporter) {
        this.location = location;
        this.reporter = reporter;
    }

    public Position getLocation() {
        return location;
    }

    public AgentID getReporter() {
        return reporter;
    }

    @Override
    public String toString() {
        return "FireReport{" + "location=" + location + ", reporter=" + reporter + '}';
    }
}

public class FireControlAgent extends Agent implements Runnable {
    private final AgentID agentID;
    private final Environment environment;
    private final Queue<Message> inbox;
    private final Queue<Action> outgoingActions;
    private AgentID closestFirefighterID = null;

    private final Queue<FireReport> pendingFirePositions;
    private final List<AgentID> availableFirefightersIDs;
    private final ConcurrentHashMap<Position, AgentID> fireReporters;

    private volatile boolean running = true;

    public FireControlAgent(AgentID agentID, Environment env, List<AgentID> availableFirefighters) {
        this.agentID = agentID;
        this.environment = env;
        this.inbox = new ConcurrentLinkedQueue<>();
        this.outgoingActions = new ConcurrentLinkedQueue<>();
        this.pendingFirePositions = new ConcurrentLinkedQueue<>();
        this.availableFirefightersIDs = availableFirefighters;
        this.fireReporters = new ConcurrentHashMap<>();
    }

    public List<AgentID> getAvailableFirefightersIDs() {
        return availableFirefightersIDs;
    }

    public Position getImminentFireLocation() {
        if (!pendingFirePositions.isEmpty()) {
            return pendingFirePositions.peek().getLocation();
        }
        return null;
    }

    @Override
    public Integer getAgentID() {
        return agentID.getId();
    }

    @Override
    public void receiveMessage(Message msg) {
        inbox.offer(msg);
    }

    @Override
    public void think(Percept p) {
        if (p instanceof FireControlPercept cp) {
            closestFirefighterID = cp.getClosestAgentID();
        }
    }

    @Override
    public Action selectAction() {
        processInbox();

        if (!pendingFirePositions.isEmpty() && !availableFirefightersIDs.isEmpty()) {
            FireReport fireReport = pendingFirePositions.peek();

            if (closestFirefighterID != null) {
                pendingFirePositions.poll();

                availableFirefightersIDs.remove(closestFirefighterID);
                DeployFirefighterPayload payload = new DeployFirefighterPayload(fireReport.getLocation());
                outgoingActions.offer(new SendMessage(new Message(agentID, closestFirefighterID, MessageType.DEPLOY_FIREFIGHTER, payload), environment));
            }
        }

        if (!outgoingActions.isEmpty()) {
            return outgoingActions.poll();
        }

        return null;
    }

    private void processInbox() {
        while (!inbox.isEmpty()) {
            Message msg = inbox.poll();
//            System.out.println(agentID + " processing message: " + msg.getType() + " from " + msg.getSender());

            switch (msg.getType()) {
                case FIRE_ALERT:
                    if (msg.getPayload() instanceof FireAlertPayload payload) {
                        Position firePosition = payload.getFirePosition();
                        AgentID reportingSensor = msg.getSender();

                        FireReport newFire = new FireReport(firePosition, reportingSensor);
                        if (!pendingFirePositions.contains(newFire)) {
                            pendingFirePositions.offer(newFire);
                            fireReporters.put(firePosition, reportingSensor);
//                            System.out.println(agentID + ": Added new pending fire alert at " + firePosition + " reported by " + reportingSensor);
                        }
                    }
                    break;
                case FIRE_EXTINGUISHED:
                    if (msg.getPayload() instanceof FireExtinguishedPayload payload) {
                        Position extinguishedPosition = payload.getExtinguishedPosition();
                        AgentID reporter = msg.getSender();

                        if (reporter != null) {
                            availableFirefightersIDs.add(reporter);
//                            System.out.println(agentID + ": " + msg.getSender() + " reported fire extinguished and is now IDLE at " + extinguishedPosition);
                            AgentID originalSensorReporter = fireReporters.remove(extinguishedPosition);
                            outgoingActions.offer(new SendMessage(new Message(agentID, originalSensorReporter, MessageType.FIRE_EXTINGUISHED, new FireExtinguishedPayload(extinguishedPosition)), environment));
                        } else {
                            System.out.println(agentID + ": Unknown firefighter agent " + msg.getSender() + " reported fire extinguished.");
                        }
                    }
                    break;
                default:
                    System.out.println(agentID + ": Received unhandled message type: " + msg.getType());
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

                Thread.sleep(100);
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
