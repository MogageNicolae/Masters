package firesystem.actions;

import agent.Action;
import agent.Agent;
import agent.Environment;
import agent.State;
import firesystem.communication.Message;

public class SendMessage extends Action {
    private final Message messageToSend;
    private final Environment environment;

    public SendMessage(Message messageToSend, Environment env) {
        this.messageToSend = messageToSend;
        this.environment = env;
    }

    public Message getMessageToSend() {
        return messageToSend;
    }

    @Override
    public State execute(Agent a, State s) {
        environment.deliverMessage(messageToSend);
        return s;
    }

    @Override
    public String toString() {
        return "SEND_MESSAGE: " + messageToSend + " to " + messageToSend.getReceiver();
    }
}
