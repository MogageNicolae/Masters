package firesystem.communication;

public class Message {
    private final AgentID sender;
    private final AgentID receiver;
    private final MessageType type;
    private final Object payload;

    public Message(AgentID sender, AgentID receiver, MessageType type, Object payload) {
        this.sender = sender;
        this.receiver = receiver;
        this.type = type;
        this.payload = payload;
    }

    public AgentID getSender() {
        return sender;
    }

    public AgentID getReceiver() {
        return receiver;
    }

    public MessageType getType() {
        return type;
    }

    public Object getPayload() {
        return payload;
    }

    @Override
    public String toString() {
        return "Message from " + sender + " to " + receiver + " [" + type + "]: " + payload;
    }
}
