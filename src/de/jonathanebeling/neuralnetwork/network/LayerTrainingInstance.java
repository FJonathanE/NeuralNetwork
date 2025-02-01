package de.jonathanebeling.neuralnetwork.network;

import de.jonathanebeling.neuralnetwork.utils.MathUtils;

public class LayerTrainingInstance {

    private double[][] costGradientW;
    private double[] costGradientB;

    private double[] lastActivations;
    private double[] lastWeightedInputs;
    private double[] lastInputs;

    private final Layer layer;

    private double[] nodeValues;

    public LayerTrainingInstance(Layer layer) {
        this.layer = layer;

        costGradientW = new double[layer.getNodesOut()][layer.getNodesIn()];
        costGradientB = new double[layer.getNodesOut()];
    }


    public double[] calculateOutputActivations(double[] inputs) {
        double[] activations = new double[layer.getNodesOut()];
        double[] weightedInputs = new double[layer.getNodesOut()];

        if (inputs.length != layer.getNodesIn()) {
            System.out.println("Input-Doubles-Array not same length as required Input-Doubles-Array length for this layer!");
        }

        for (int out = 0; out < layer.getNodesOut(); out++) {

            double weightedInput = MathUtils.sumMultipliedArrays(inputs, layer.getWeightsIn()[out]);
            weightedInput += layer.getBiases()[out];


            weightedInputs[out] = weightedInput;
            activations[out] = layer.getActivationFunction().activation(weightedInput);
        }

        lastActivations = activations;
        lastWeightedInputs = weightedInputs;
        lastInputs = inputs;

        return activations;
    }


    public void updateGradients() {
        for (int nodeOut = 0; nodeOut < layer.getNodesOut(); nodeOut++) {
            for (int nodeIn = 0; nodeIn < layer.getNodesIn(); nodeIn++) {
                double derivativeCostWrtWeight = lastInputs[nodeIn] * nodeValues[nodeOut];

                costGradientW[nodeOut][nodeIn] += derivativeCostWrtWeight;
            }



            double derivativeCostWrtBias = 1 * nodeValues[nodeOut];
            costGradientB[nodeOut] += derivativeCostWrtBias;
        }
    }


    public void updateOutputLayerNodeValues(double[] expectedOutputs) {
        double[] nodeValues = new double[expectedOutputs.length];

        for (int i = 0; i < nodeValues.length; i++) {
            // Evaluate partial derivatives for current node: cost/activation & activation/weightedInput

            double costDerivative = layer.getCostFunction().derivative(lastActivations[i], expectedOutputs[i]);
            double activationDerivative = layer.getActivationFunction().derivative(lastWeightedInputs[i]);
            nodeValues[i] = activationDerivative * costDerivative;
        }

        this.nodeValues = nodeValues;
    }

    public void updateHiddenLayerNodeValues(LayerTrainingInstance oldLayer) {
        double[] newNodeValues = new double[layer.getNodesOut()];
        double[] oldNodeValues = oldLayer.getNodeValues();

        for (int newNodeIndex = 0; newNodeIndex < newNodeValues.length; newNodeIndex++) {
            double newNodeValue = 0;
            for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.length; oldNodeIndex++) {
                double weightedInputDerivative = oldLayer.getLayer().getWeightsIn()[oldNodeIndex][newNodeIndex];
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }


            newNodeValue *= layer.getActivationFunction().derivative(lastWeightedInputs[newNodeIndex]);
            newNodeValues[newNodeIndex] = newNodeValue;
        }

        this.nodeValues = newNodeValues;
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

    public double[] getNodeValues() {
        return nodeValues;
    }
}
