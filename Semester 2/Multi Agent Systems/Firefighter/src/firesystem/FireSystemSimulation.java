package firesystem;

import agent.Agent;
import agent.Simulation;
import firesystem.communication.AgentID;
import firesystem.firecontrol.FireControlAgent;
import firesystem.firefighter.FirefighterAgent;
import firesystem.firesensor.FireSensorAgent;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FireSystemSimulation extends Simulation {
    public FireSystemSimulation(FireSystemEnvironment environment, List<? extends Agent> agents, Map<AgentID, Agent> registeredAgents) {
        super(environment, agents, registeredAgents);
    }

    @Override
    protected boolean isComplete() {
        return ((FireSystemState) env.currentState()).getFireCount() == 0;
    }

    public static void main(String[] args) {
        System.out.println("Fire System Simulation");
        System.out.println("-----------------------");
        System.out.println();

        FireSystemEnvironment environment = new FireSystemEnvironment();
        FireSystemState initialState = FireSystemState.getInitState();

        List<Agent> agents = new java.util.ArrayList<>();
        List<AgentID> availableAgentIDs = new java.util.ArrayList<>();
        Map<AgentID, Agent> agentMap = new HashMap<>();

        int numberOfFireSensors = initialState.getNumberOfFireSensors();
        int numberOfFirefighters = initialState.getNumberOfFirefighters();

        for (int i = 0; i < numberOfFireSensors; i++) {
            AgentID agentID = new AgentID(i, "FSA");
            FireSensorAgent fireSensorAgent = new FireSensorAgent(agentID, environment);

            agents.add(fireSensorAgent);
            agentMap.put(agentID, fireSensorAgent);
        }

        for (int i = 0; i < numberOfFirefighters; i++) {
            AgentID agentID = new AgentID(i, "FFA");
            FirefighterAgent firefighterAgent = new FirefighterAgent(agentID, environment);

            agents.add(firefighterAgent);
            availableAgentIDs.add(agentID);
            agentMap.put(agentID, firefighterAgent);
        }

        initialState.placeFirefighters(availableAgentIDs);

        AgentID fireControlAgentID = new AgentID(0, "FCA");
        FireControlAgent fireControlAgent = new FireControlAgent(fireControlAgentID, environment, availableAgentIDs);

        agents.add(fireControlAgent);
        agentMap.put(fireControlAgentID, fireControlAgent);

        FireSystemSimulation simulation = new FireSystemSimulation(environment, agents, agentMap);

        simulation.start(initialState);
    }
}
