package firesystem.actions;

import agent.Action;
import agent.Agent;
import agent.State;
import firesystem.FireSystemState;
import firesystem.firefighter.FirefighterAgent;

public class ExtinguishFire extends Action {
    public ExtinguishFire() {}

    @Override
    public State execute(Agent a, State s) {
        FireSystemState fireState = (FireSystemState) s;
        FirefighterAgent firefighterAgent = (FirefighterAgent) a;

        fireState.extinguishFire(firefighterAgent.getAgentID());
        return fireState;
    }

    @Override
    public String toString() {
        return "Extinguish Fire";
    }
}
