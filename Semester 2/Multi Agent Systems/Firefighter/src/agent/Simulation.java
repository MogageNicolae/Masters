package agent;

import firesystem.communication.AgentID;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public abstract class Simulation {
    protected List<? extends Agent> agents = null;
    Map<AgentID, Agent> registeredAgents = null;
    protected Environment env;
    private final Lock environmentLock = new ReentrantLock();
    private volatile boolean simulationRunning = true;

    public Simulation(Environment e, List<? extends Agent> a, Map<AgentID, Agent> rA) {
        registeredAgents = rA;
        agents = a;
        env = e;
    }

    public void start(State initState) {
        env.setInitialState(initState);
        env.currentState().display();

        ExecutorService executor = Executors.newFixedThreadPool(agents.size());

        for (Agent agent : agents) {
            if (agent instanceof Runnable) {
                executor.submit((Runnable) agent);
            }
        }

        new Thread(() -> {
            while (simulationRunning) {
                try {
                    Thread.sleep(100);
                    environmentLock.lock();
                    try {
                        env.processMessages(registeredAgents);
                        if (isComplete()) {
                            simulationRunning = false;
                            for (Agent agent : agents) {
                                if (agent instanceof firesystem.firefighter.FirefighterAgent) {
                                    ((firesystem.firefighter.FirefighterAgent) agent).stopRunning();
                                } else if (agent instanceof firesystem.firesensor.FireSensorAgent) {
                                    ((firesystem.firesensor.FireSensorAgent) agent).stopRunning();
                                } else if (agent instanceof firesystem.firecontrol.FireControlAgent) {
                                    ((firesystem.firecontrol.FireControlAgent) agent).stopRunning();
                                }
                            }
                        }
//                        env.currentState().display();
                    } finally {
                        environmentLock.unlock();
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    System.err.println("Simulation termination thread interrupted.");
                }
            }
        }).start();

        executor.shutdown();
        try {
            if (!executor.awaitTermination(120, TimeUnit.SECONDS)) {
                executor.shutdownNow();
                if (!executor.awaitTermination(10, TimeUnit.SECONDS)) {
                    System.err.println("Executor did not terminate.");
                }
            }
        } catch (InterruptedException ie) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        env.currentState().display();
        System.out.println("END of simulation");
    }

    /**
     * Is the simulation over? Returns true if it is, otherwise false.
     */
    protected abstract boolean isComplete();
}
