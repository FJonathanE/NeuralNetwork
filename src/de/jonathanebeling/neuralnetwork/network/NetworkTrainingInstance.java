package de.jonathanebeling.neuralnetwork.network;

import de.jonathanebeling.neuralnetwork.data.DataPoint;

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



    public void applyAllGradients(double learningRate) {

        for (LayerTrainingInstance layer : layerTrainingInstances) {
            layer.getLayer().applyGradients(learningRate, layer);
        }
    }

    protected double trainOnDatapoint(DataPoint datapoint) {
        double[] outputs = calculateOutputs(datapoint.getInputActivation());

        LayerTrainingInstance outputLayer = layerTrainingInstances[layerTrainingInstances.length - 1];
        outputLayer.updateOutputLayerNodeValues(datapoint.getExpectedOutputActivation());
        outputLayer.updateGradients();


        for (int hiddenLayerIndex = layerTrainingInstances.length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--) {
            LayerTrainingInstance hiddenLayer = layerTrainingInstances[hiddenLayerIndex];
            hiddenLayer.updateHiddenLayerNodeValues(layerTrainingInstances[hiddenLayerIndex + 1]);
            hiddenLayer.updateGradients();
        }

        return network.getCostFunction().dataPointCost(datapoint, outputs);
    }
}
