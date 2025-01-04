package network;

public class NetworkTrainingInstance {

    private final NeuralNetwork network;
    private final LayerTrainingInstance[] layerTrainingInstances;


    public NetworkTrainingInstance(NeuralNetwork network) {
        this.network = network;

        layerTrainingInstances = new LayerTrainingInstance[network.getLayers().length];

        for (int i = 0; i < layerTrainingInstances.length; i++) {
            layerTrainingInstances[i] = new LayerTrainingInstance(network.getLayers()[i]);
        }


    }


}
