package agent;

import firesystem.communication.Message;

/**
 * An abstract software agent class. The agent must be managed by the Simulation
 * class
 */
public abstract class Agent {
    public abstract Integer getAgentID();
    public abstract void receiveMessage(Message msg);
    public abstract void think(Percept p);
    public abstract Action selectAction();
}
