package firesystem.firecontrol;

import agent.Agent;
import agent.Percept;
import agent.State;
import firesystem.FireSystemState;
import firesystem.communication.AgentID;
import firesystem.utils.Position;

public class FireControlPercept extends Percept {
    private final AgentID closestAgentID;

    public FireControlPercept(State s, Agent a) {
        super(s, a);

        FireSystemState state = (FireSystemState) s;
        FireControlAgent fireControlAgent = (FireControlAgent) a;

        AgentID closestAgent = null;
        double minDistance = Double.MAX_VALUE;
        Position fireLocation = fireControlAgent.getImminentFireLocation();

        if (fireLocation == null) {
            closestAgentID = null;
            return;
        }

        for (AgentID aId : fireControlAgent.getAvailableFirefightersIDs()) {
            double distance = calculateManhattanDistance(state.getCurrentPosition(aId.getId()), fireLocation);
            if (distance < minDistance) {
                minDistance = distance;
                closestAgent = aId;
            }
        }
        closestAgentID = closestAgent;
    }

    private double calculateManhattanDistance(Position p1, Position p2) {
        return Math.abs(p1.getX() - p2.getY()) + Math.abs(p1.getY() - p2.getY());
    }

    public AgentID getClosestAgentID() {
        return closestAgentID;
    }

    @Override
    public String toString() {
        return "";
    }
}
