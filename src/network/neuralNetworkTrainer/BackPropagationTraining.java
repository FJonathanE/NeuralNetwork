package network.neuralNetworkTrainer;

import network.DataPoint;
import network.Layer;
import network.NeuralNetwork;

public class BackPropagationTraining extends TrainingAlgorithm {


    @Override
    protected double updateAllGradients(DataPoint datapoint, NeuralNetwork network) {
        double[] outputs = network.calculateOutputs(datapoint.getInputActivation());

        Layer[] layers = network.getLayers();
        Layer outputLayer = layers[layers.length - 1];
        double[] nodeValues = outputLayer.calculateOutputLayerNodeValues(datapoint.getExpectedOutputActivation());
        outputLayer.updateGradients(nodeValues);


        for (int hiddenLayerIndex = layers.length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--) {
            Layer hiddenLayer = layers[hiddenLayerIndex];
            nodeValues = hiddenLayer.calculateHiddenLayerNodeValues(layers[hiddenLayerIndex + 1], nodeValues);
            hiddenLayer.updateGradients(nodeValues);
        }

        return network.getCostFunction().dataPointCost(datapoint, outputs);
    }


}
