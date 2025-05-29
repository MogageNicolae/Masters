package firesystem.firefighter;

import agent.Agent;
import agent.Percept;
import agent.State;
import firesystem.FireSystemState;
import firesystem.utils.Position;

public class FirefighterPercept extends Percept {
    private final boolean fireDetected;
    private final Position nextPosition;

    public FirefighterPercept(State s, Agent a) {
        super(s, a);

        FireSystemState state = (FireSystemState) s;
        FirefighterAgent firefighterAgent = (FirefighterAgent) a;

        Position targetFire = firefighterAgent.getTargetFirePosition();
        if (targetFire != null) {
            nextPosition = state.getNextPosition(firefighterAgent.getAgentID(), targetFire);
        } else {
            nextPosition = null;
        }
        fireDetected = state.isFireDetected(firefighterAgent.getAgentID(), targetFire);
    }

    public boolean isFireDetected() {
        return fireDetected;
    }

    public Position getNextPosition() {
        return nextPosition;
    }

    @Override
    public String toString() {
        return "";
    }
}
