package agent;


import firesystem.communication.AgentID;
import firesystem.communication.Message;

import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;

public abstract class Environment {
    private final ConcurrentLinkedQueue<Message> messageQueue;
    protected State state;

    public Environment() {
        messageQueue = new ConcurrentLinkedQueue<>();
    }

    public abstract Percept getPercept(Agent a);

    public void updateState(Agent a, Action action) {
        state = action.execute(a, state);
    }

    public State currentState() {
        return state;
    }

    public void setInitialState(State s) {
        state = s;
    }

    public void deliverMessage(Message messageToSend) {
        messageQueue.offer(messageToSend);
    }

    public void processMessages(Map<AgentID, Agent> registeredAgents) {
        while (!messageQueue.isEmpty()) {
            Message msg = messageQueue.poll();
            Agent receiverAgent = registeredAgents.get(msg.getReceiver());
            if (receiverAgent != null) {
                receiverAgent.receiveMessage(msg);
            } else {
                System.err.println("Environment: Message to unknown agent: " + msg.getReceiver());
            }
        }
    }
}



