package firesystem.communication;

public class AgentID {
    private final Integer id;
    private final String type;

    public AgentID(Integer id, String type) {
        this.id = id;
        this.type = type;
    }

    public Integer getId() {
        return id;
    }

    public String getType() {
        return type;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof AgentID)) return false;
        AgentID agentID = (AgentID) o;
        return id.equals(agentID.id) && type.equals(agentID.type);
    }

    @Override
    public int hashCode() {
        int result = id.hashCode();
        result = 31 * result + type.hashCode();
        return result;
    }

    @Override
    public String toString() {
        return "AgentID{" +
                "id='" + id + '\'' +
                ", type='" + type + '\'' +
                '}';
    }
}
