package network;

public class LayerTrainingInstance {

    private double[][] costGradientW;
    private double[] costGradientB;

    private double[] lastActivations;
    private double[] lastWeightedInputs;
    private double[] lastInputs;

    private final Layer layer;

    public LayerTrainingInstance(Layer layer) {
        this.layer = layer;
    }

    public void initializeGradients() {
        costGradientW = new double[layer.getNodesIn()][layer.getNodesOut()];
        costGradientB = new double[layer.getNodesOut()];
    }
}
