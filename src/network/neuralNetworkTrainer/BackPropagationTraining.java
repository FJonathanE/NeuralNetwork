package network.neuralNetworkTrainer;

import network.*;

public class BackPropagationTraining extends TrainingAlgorithm {


    @Override
    protected double updateAllGradients(DataPoint datapoint, NetworkTrainingInstance network) {
        double[] outputs = network.calculateOutputs(datapoint.getInputActivation());

        LayerTrainingInstance[] layers = network.getLayerTrainingInstances();
        LayerTrainingInstance outputLayer = layers[layers.length - 1];
        double[] nodeValues = calculateOutputLayerNodeValues(datapoint.getExpectedOutputActivation(), outputLayer);
        updateGradients(nodeValues, outputLayer);


        for (int hiddenLayerIndex = layers.length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--) {
            LayerTrainingInstance hiddenLayer = layers[hiddenLayerIndex];
            nodeValues = calculateHiddenLayerNodeValues(hiddenLayer, layers[hiddenLayerIndex + 1], nodeValues);
            updateGradients(nodeValues, hiddenLayer);
        }

        return network.getNetwork().getCostFunction().dataPointCost(datapoint, outputs);
    }


    public void updateGradients(double[] nodeValues, LayerTrainingInstance layer) {
        for (int nodeOut = 0; nodeOut < layer.getLayer().getNodesOut(); nodeOut++) {
            for (int nodeIn = 0; nodeIn < layer.getLayer().getNodesIn(); nodeIn++) {
                double derivativeCostWrtWeight = layer.getLastInputs()[nodeIn] * nodeValues[nodeOut];

                layer.addToCostGradientW(derivativeCostWrtWeight, nodeIn, nodeOut);
            }

            double derivativeCostWrtBias = 1 * nodeValues[nodeOut];
            layer.addToCostGradientB(derivativeCostWrtBias, nodeOut);
        }
    }


    public double[] calculateOutputLayerNodeValues(double[] expectedOutputs, LayerTrainingInstance trainingInstance) {
        double[] nodeValues = new double[expectedOutputs.length];

        for (int i = 0; i < nodeValues.length; i++) {
            // Evaluate partial derivatives for current node: cost/activation & activation/weightedInput

            double costDerivative = trainingInstance.getLayer().getCostFunction().nodeCostDerivative(trainingInstance.getLastActivations()[i], expectedOutputs[i]);
            double activationDerivative = trainingInstance.getLayer().getActivationFunction().derivative(trainingInstance.getLastWeightedInputs()[i]);
            nodeValues[i] = activationDerivative * costDerivative;
        }

        return nodeValues;
    }

    public double[] calculateHiddenLayerNodeValues(LayerTrainingInstance currentLayer, LayerTrainingInstance oldLayer, double[] oldNodeValues) {
        double[] newNodeValues = new double[currentLayer.getLayer().getNodesOut()];

        for (int newNodeIndex = 0; newNodeIndex < newNodeValues.length; newNodeIndex++) {
            double newNodeValue = 0;
            for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.length; oldNodeIndex++) {
                double weightedInputDerivative = oldLayer.getLayer().getWeightsIn()[newNodeIndex][oldNodeIndex];
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }

            newNodeValue *= currentLayer.getLayer().getActivationFunction().derivative(currentLayer.getLastWeightedInputs()[newNodeIndex]);
            newNodeValues[newNodeIndex] = newNodeValue;
        }

        return newNodeValues;
    }


}
