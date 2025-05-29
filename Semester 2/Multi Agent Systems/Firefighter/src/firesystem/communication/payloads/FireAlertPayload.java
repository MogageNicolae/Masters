package firesystem.communication.payloads;

import firesystem.utils.Position;

public class FireAlertPayload {
    private final Position firePosition;

    public FireAlertPayload(Position firePosition) {
        this.firePosition = firePosition;
    }

    public Position getFirePosition() {
        return firePosition;
    }
}
