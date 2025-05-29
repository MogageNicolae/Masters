package firesystem.actions;

import agent.Action;
import agent.Agent;
import agent.State;
import firesystem.FireSystemState;
import firesystem.firefighter.FirefighterAgent;
import firesystem.utils.Position;

public class Move extends Action {
    private final Position nextPosition;

    public Move(Position nextPosition) {
        this.nextPosition = nextPosition;
    }

    @Override
    public State execute(Agent a, State s) {
        FireSystemState fireState = (FireSystemState) s;
        FirefighterAgent firefighterAgent = (FirefighterAgent) a;

        fireState.setFirefighterAgentPosition(firefighterAgent.getAgentID(), nextPosition);
        return fireState;
    }

    @Override
    public String toString() {
        return "Move Firefighter";
    }
}
