package firesystem.communication.payloads;

import firesystem.communication.AgentID;
import firesystem.utils.Position;

public class FireExtinguishedPayload {
    private final Position extinguishedPosition;

    public FireExtinguishedPayload(Position extinguishedPosition) {
        this.extinguishedPosition = extinguishedPosition;
    }

    public Position getExtinguishedPosition() {
        return extinguishedPosition;
    }
}
