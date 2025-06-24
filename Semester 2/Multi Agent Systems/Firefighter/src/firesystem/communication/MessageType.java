package firesystem.communication;

public enum MessageType {
    FIRE_ALERT, // FSA to FCA with the position of the fire
    DEPLOY_FIREFIGHTER, // FCA to FFA with the position of the fire to extinguish
    FIRE_EXTINGUISHED, // FFA to FCA to confirm the fire has been extinguished, with the position of the agent and its id
}
