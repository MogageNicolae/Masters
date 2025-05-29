package firesystem;

import agent.Action;
import agent.Agent;
import agent.Environment;
import agent.Percept;
import firesystem.firecontrol.FireControlAgent;
import firesystem.firecontrol.FireControlPercept;
import firesystem.firefighter.FirefighterAgent;
import firesystem.firefighter.FirefighterPercept;
import firesystem.firesensor.FireSensorAgent;
import firesystem.firesensor.FireSensorPercept;

public class FireSystemEnvironment extends Environment {
    public FireSystemEnvironment() {
        super();
    }

    @Override
    public Percept getPercept(Agent a) {
        switch (a) {
            case FireSensorAgent fireSensorAgent -> {
                return new FireSensorPercept(state, a);
            }
            case FirefighterAgent firefighterAgent -> {
                return new FirefighterPercept(state, a);
            }
            case FireControlAgent fireControlAgent -> {
                return new FireControlPercept(state, a);
            }
            case null, default -> {
                System.out.println("Unknown agent type for percept creation.");
                return null;
            }
        }
    }

    @Override
    public void updateState(Agent a, Action action) {
        super.updateState(a, action);
    }
}
