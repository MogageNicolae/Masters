import jade.core.Profile;
import jade.core.ProfileImpl;
import jade.core.Runtime;
import jade.wrapper.AgentController;
import jade.wrapper.ContainerController;

public class Main {
    public static void main(String[] args) {
        Runtime rt = Runtime.instance();
        Profile p = new ProfileImpl();
        ContainerController cc = rt.createMainContainer(p);

        try {
            AgentController tsa1 = cc.createNewAgent("TrafficSensorAgent1", "agents.TrafficSensorAgent", new Object[]{"stage_1.txt"});
            AgentController tsa2 = cc.createNewAgent("TrafficSensorAgent2", "agents.TrafficSensorAgent", new Object[]{"stage_2.txt"});

            AgentController tca = cc.createNewAgent("TrafficControlAgent", "agents.TrafficControlAgent", null);

            AgentController tla1 = cc.createNewAgent("TrafficLightAgent1", "agents.TrafficLightAgent", null);
            AgentController tla2 = cc.createNewAgent("TrafficLightAgent2", "agents.TrafficLightAgent", null);

            tsa1.start();
            tsa2.start();
            tca.start();
            tla1.start();
            tla2.start();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
