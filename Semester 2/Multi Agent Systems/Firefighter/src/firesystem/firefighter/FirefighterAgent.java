package firesystem.firefighter;

import agent.Action;
import agent.Agent;
import agent.Environment;
import agent.Percept;
import firesystem.actions.ExtinguishFire;
import firesystem.actions.Move;
import firesystem.actions.SendMessage;
import firesystem.communication.AgentID;
import firesystem.communication.Message;
import firesystem.communication.MessageType;
import firesystem.communication.payloads.DeployFirefighterPayload;
import firesystem.communication.payloads.FireExtinguishedPayload;
import firesystem.utils.Position;

import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

public class FirefighterAgent extends Agent implements Runnable {
    private final AgentID agentID;
    private final Environment environment;
    private final Queue<Message> inbox;
    private final Queue<Action> outgoingActions;
    private boolean isFireDetected = false;

    private Position currentTarget;
    private Position nextPosition;

    private volatile boolean running = true;

    public FirefighterAgent(AgentID agentID, Environment env) {
        this.agentID = agentID;
        this.environment = env;
        this.inbox = new ConcurrentLinkedQueue<>();
        this.outgoingActions = new ConcurrentLinkedQueue<>();
    }

    @Override
    public Integer getAgentID() {
        return agentID.getId();
    }

    public Position getTargetFirePosition() {
        return currentTarget;
    }

    @Override
    public void receiveMessage(Message msg) {
        inbox.offer(msg);
    }

    @Override
    public void think(Percept p) {
        FirefighterPercept fp = (FirefighterPercept) p;

        isFireDetected = fp.isFireDetected();
        nextPosition = fp.getNextPosition();
    }

    @Override
    public Action selectAction() {
        processInbox();

        if (!outgoingActions.isEmpty()) {
            return outgoingActions.poll();
        }

        if (isFireDetected) {
            FireExtinguishedPayload payload = new FireExtinguishedPayload(currentTarget);
            Message extinguishMessage = new Message(agentID, new AgentID(0, "FCA"), MessageType.FIRE_EXTINGUISHED, payload);
            outgoingActions.offer(new SendMessage(extinguishMessage, environment));

            currentTarget = null;
            nextPosition = null;

            return new ExtinguishFire();
        }

        if (currentTarget != null && nextPosition != null) {
            return new Move(nextPosition);
        }

        return null;
    }

    private void processInbox() {
        while (!inbox.isEmpty()) {
            Message msg = inbox.poll();
//            System.out.println(agentID + " processing message: " + msg.getType() + " from " + msg.getSender());

            if (msg.getType() == MessageType.DEPLOY_FIREFIGHTER) {
                if (msg.getPayload() instanceof DeployFirefighterPayload payload) {
                    currentTarget = payload.getTargetFirePosition();
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
//                Percept p =
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
