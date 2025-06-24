package firesystem.communication.payloads;

import firesystem.utils.Position;

public class DeployFirefighterPayload {
    private final Position targetFirePosition;

    public DeployFirefighterPayload(Position targetFirePosition) {
        this.targetFirePosition = targetFirePosition;
    }

    public Position getTargetFirePosition() {
        return targetFirePosition;
    }
}
