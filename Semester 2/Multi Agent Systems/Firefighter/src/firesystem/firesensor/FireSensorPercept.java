package firesystem.firesensor;

import agent.Agent;
import agent.Percept;
import agent.State;
import firesystem.FireSystemState;
import firesystem.utils.Position;

import java.util.List;

public class FireSensorPercept extends Percept {
    private final List<Position> firePositions;

    public FireSensorPercept(State s, Agent a) {
        super(s, a);

        FireSensorAgent fireSensorAgent = (FireSensorAgent) a;
        FireSystemState state = (FireSystemState) s;
        this.firePositions = new java.util.ArrayList<>();

        int startRow = (fireSensorAgent.getAgentID() * (state.getHeight() / state.getNumberOfFireSensors()));
        int endRow = startRow + (state.getHeight() / state.getNumberOfFireSensors()) - 1;
        int gridWidth = state.getWidth();

        for (int i = startRow; i <= endRow; i++) {
            for (int j = 0; j < gridWidth; j++) {
                if (state.isFireAt(i, j)) {
                    this.firePositions.add(new Position(i, j));
                }
            }
        }
    }

    public List<Position> getFirePositions() {
        return firePositions;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("FireSensorPercept: ");
        for (Position pos : this.firePositions) {
            sb.append("Fire at ").append(pos).append("; ");
        }
        return sb.toString();
    }
}
