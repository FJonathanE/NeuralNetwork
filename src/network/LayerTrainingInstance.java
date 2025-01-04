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

        costGradientW = new double[layer.getNodesIn()][layer.getNodesOut()];
        costGradientB = new double[layer.getNodesOut()];
    }


    public void addToCostGradientW(double deltaCostGradientW, int inNodeIndex, int outNodeIndex){
        costGradientW[inNodeIndex][outNodeIndex] += deltaCostGradientW;
    }

    public void addToCostGradientB(double deltaCostGradientB, int outNodeIndex){
        costGradientB[outNodeIndex] += deltaCostGradientB;
    }

    public double[] calculateOutputActivations(double[] inputs) {
        double[] activations = new double[layer.getNodesOut()];
        double[] weightedInputs = new double[layer.getNodesOut()];

        if (inputs.length != layer.getNodesIn()) {
            System.out.println("Input-Doubles-Array not same length as required Input-Doubles-Array length for this layer!");
        }

        for (int out = 0; out < layer.getNodesOut(); out++) {
            double weightedInput = 0;

            for (int in = 0; in < inputs.length; in++) {
                weightedInput += inputs[in] * layer.getWeightsIn()[in][out];
            }


            weightedInput += layer.getBiases()[out];
            weightedInputs[out] = weightedInput;
            activations[out] = layer.getActivationFunction().activation(weightedInput);
        }

        lastActivations = activations;
        lastWeightedInputs = weightedInputs;
        lastInputs = inputs;

        return activations;
    }



    public Layer getLayer() {
        return layer;
    }

    public double[] getLastActivations() {
        return lastActivations;
    }

    public double[] getLastWeightedInputs() {
        return lastWeightedInputs;
    }

    public double[] getLastInputs() {
        return lastInputs;
    }

    public double[][] getCostGradientW() {
        return costGradientW;
    }

    public double[] getCostGradientB() {
        return costGradientB;
    }
}
