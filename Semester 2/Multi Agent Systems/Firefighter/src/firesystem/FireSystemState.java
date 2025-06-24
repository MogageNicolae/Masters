package firesystem;

import agent.State;
import firesystem.communication.AgentID;
import firesystem.utils.Position;

import java.util.*;
import java.util.Map;

public class FireSystemState extends State {
    protected static final int CLEAR = 0;
    protected static final int FIRE = 1;
    protected static final int WALL = 2;
    protected static final int OBSTACLE = 3;

    protected static final int DEFAULT_HEIGHT = 10;
    protected static final int DEFAULT_WIDTH = 10;
    protected static final int DEFAULT_NUMBER_OF_FIRE_SENSORS = 3;
    protected static final int DEFAULT_NUMBER_OF_FIRE_FIGHTERS = 2;
    protected static int[][] defaultMap = {
        {WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL},
        {WALL, CLEAR, CLEAR, FIRE, CLEAR, CLEAR, CLEAR, FIRE, CLEAR, WALL},
        {WALL, CLEAR, OBSTACLE, CLEAR, CLEAR, FIRE, CLEAR, OBSTACLE, CLEAR, WALL},
        {WALL, FIRE, CLEAR, CLEAR, OBSTACLE, CLEAR, CLEAR, FIRE, CLEAR, WALL},
        {WALL, CLEAR, FIRE, OBSTACLE, CLEAR, CLEAR, FIRE, OBSTACLE, CLEAR, WALL},
        {WALL, OBSTACLE, CLEAR, FIRE, CLEAR, OBSTACLE, CLEAR, FIRE, CLEAR, WALL},
        {WALL, FIRE, CLEAR, OBSTACLE, FIRE, CLEAR, OBSTACLE, CLEAR, FIRE, WALL},
        {WALL, CLEAR, OBSTACLE, FIRE, CLEAR, OBSTACLE, FIRE, CLEAR, OBSTACLE ,WALL},
        {WALL ,FIRE ,CLEAR ,CLEAR ,OBSTACLE ,FIRE ,CLEAR ,CLEAR ,FIRE ,WALL},
        {WALL ,WALL ,WALL ,WALL ,WALL ,WALL ,WALL ,WALL ,WALL ,WALL}
    };
//    protected static int[][] defaultMap = {
//        {WALL, WALL, WALL, WALL, WALL, WALL},
//        {WALL, CLEAR, CLEAR, FIRE, CLEAR, WALL},
//        {WALL, CLEAR, OBSTACLE, CLEAR, CLEAR, WALL},
//        {WALL, FIRE, CLEAR, CLEAR, OBSTACLE, WALL},
//        {WALL, CLEAR, FIRE, OBSTACLE, CLEAR, WALL},
//        {WALL, OBSTACLE, CLEAR, FIRE, CLEAR, WALL},
//        {WALL, FIRE, CLEAR, OBSTACLE, FIRE, WALL},
//        {WALL, CLEAR, OBSTACLE, FIRE, CLEAR ,WALL},
//        {WALL ,FIRE ,CLEAR ,CLEAR ,OBSTACLE ,WALL},
//        {WALL ,WALL ,WALL ,WALL ,WALL ,WALL}
//    };

    protected int[][] map;

    protected int height;
    protected int width;
    protected int numberOfFireSensors;
    protected int numberOfFirefighters;

    protected Map<Integer, Position> firefighterPositions = new HashMap<>();

    public static FireSystemState getInitState() {
        FireSystemState state = new FireSystemState();

        state.width = DEFAULT_WIDTH;
        state.height = DEFAULT_HEIGHT;
        state.numberOfFireSensors = DEFAULT_NUMBER_OF_FIRE_SENSORS;
        state.numberOfFirefighters = DEFAULT_NUMBER_OF_FIRE_FIGHTERS;
        state.map = new int[state.height][state.width];
        state.map = defaultMap.clone();
        return state;
    }

    public void placeFirefighters(List<AgentID> availableAgentIDs) {
        for (AgentID agentID : availableAgentIDs) {
            int x, y;
            do {
                x = (int) (Math.random() * DEFAULT_HEIGHT);
                y = (int) (Math.random() * DEFAULT_WIDTH);
            } while (defaultMap[x][y] != CLEAR && !firefighterPositions.containsValue(new Position(x, y)));

            Position position = new Position(x, y);
            this.firefighterPositions.put(agentID.getId(), position);
        }
    }

    public FireSystemState() {
    }

    @Override
    public void display() {
        System.out.println("Fire System State:");

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                System.out.print(" | ");
                if (firefighterPositions.containsValue(new Position(i, j))) {
                    System.out.print("A");
                    continue;
                }
                switch (map[i][j]) {
                    case CLEAR -> System.out.print(" ");
                    case FIRE -> System.out.print("F");
                    case WALL -> System.out.print("#");
                    case OBSTACLE -> System.out.print("O");
                }
            }
            System.out.println();
        }
    }

    public Position getNextPosition(Integer agentID, Position targetFire) {
        Position currentPosition = firefighterPositions.get(agentID);
        if (currentPosition == null) {
            return null;
        }

        return findNextPositionOnShortestPath(currentPosition, targetFire);
    }

    private Position findNextPositionOnShortestPath(Position start, Position target) {
        if (start.equals(target)) {
            return start;
        }

        Queue<Position> queue = new LinkedList<>();
        Map<Position, Position> parentMap = new HashMap<>();
        Set<Position> visited = new HashSet<>();

        queue.add(start);
        visited.add(start);

        int[] dx = {-1, 1, 0, 0}; // Up, Down, Left, Right
        int[] dy = {0, 0, -1, 1};

        while (!queue.isEmpty()) {
            Position current = queue.poll();

            if (current.equals(target)) {
                return reconstructPath(start, target, parentMap);
            }

            for (int i = 0; i < 4; i++) {
                int newX = current.getX() + dx[i];
                int newY = current.getY() + dy[i];
                Position neighbor = new Position(newX, newY);

                if (isValidPosition(newX, newY) && !visited.contains(neighbor)) {
                    queue.add(neighbor);
                    visited.add(neighbor);
                    parentMap.put(neighbor, current);
                }
            }
        }
        return null;
    }

    private Position reconstructPath(Position start, Position target, Map<Position, Position> parentMap) {
        LinkedList<Position> path = new LinkedList<>();
        Position current = target;
        while (current != null && !current.equals(start)) {
            path.addFirst(current);
            current = parentMap.get(current);
        }
        if (path.isEmpty()) {
            return start;
        }
        return path.getFirst();
    }


    private boolean isValidPosition(int x, int y) {
        return map[x][y] != WALL && map[x][y] != OBSTACLE;
    }

    public boolean isFireDetected(Integer agentID, Position targetFire) {
        Position position = firefighterPositions.get(agentID);
        if (position == null) {
            return false;
        }
        int x = position.getX();
        int y = position.getY();

        if (position != targetFire) {
            return false;
        }

        return isFireAt(x, y);
    }

    public Position getCurrentPosition(Integer id) {
        return firefighterPositions.get(id);
    }

    public int getHeight() {
        return height;
    }

    public int getNumberOfFireSensors() {
        return numberOfFireSensors;
    }

    public int getNumberOfFirefighters() {
        return numberOfFirefighters;
    }

    public int getWidth() {
        return width;
    }

    public boolean isFireAt(int i, int j) {
//        display();
        return map[i][j] == FIRE;
    }

    public void setFirefighterAgentPosition(Integer agentID, Position nextPosition) {
        firefighterPositions.put(agentID, nextPosition);
    }

    public void extinguishFire(Integer agentID) {
        Position position = firefighterPositions.get(agentID);
        System.out.println("Extinguishing fire at " + position);
        map[position.getX()][position.getY()] = CLEAR;
//        display();
    }

    public int getFireCount() {
        int count = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (map[i][j] == FIRE) {
                    count++;
                }
            }
        }
        return count;
    }
}
