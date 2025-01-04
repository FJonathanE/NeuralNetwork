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


    public double[] calculateOutputs(double[] inputActivations) {
        for (LayerTrainingInstance layer : layerTrainingInstances) {
            inputActivations = layer.calculateOutputActivations(inputActivations);
        }
        return inputActivations;
    }

    public NeuralNetwork getNetwork() {
        return network;
    }

    public LayerTrainingInstance[] getLayerTrainingInstances() {
        return layerTrainingInstances;
    }

    public void applyAllGradients(double learningRate) {

        for (LayerTrainingInstance layer : layerTrainingInstances) {
            layer.getLayer().applyGradients(learningRate, layer);
        }

    }
}
