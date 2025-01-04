package network;

import network.activationFunctions.ActivationFunction;
import network.costFunctions.CostFunction;

import java.io.Serial;
import java.io.Serializable;
import java.util.Random;

public class Layer implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    private final int nodesIn;
    private final int nodesOut;

    private double[][] weightsIn;
    private double[] biases;

    private transient double[][] costGradientW;
    private transient double[] costGradientB;

    private transient double[] lastActivations;
    private transient double[] lastWeightedInputs;
    private transient double[] lastInputs;

    private final ActivationFunction activationFunction;
    private final CostFunction costFunction;


    public Layer(int nodesIn, int nodesOut, Random random, ActivationFunction activationFunction, CostFunction costFunction) {
        this.nodesIn = nodesIn;
        this.nodesOut = nodesOut;
        this.activationFunction = activationFunction;
        this.costFunction = costFunction;

        weightsIn = new double[nodesIn][nodesOut];
        biases = new double[nodesOut];


        //Random weights and biases on initialization
        for (int i = 0; i < nodesOut; i++) {
            for (int j = 0; j < nodesIn; j++) {
                weightsIn[j][i] = random.nextGaussian() * Math.sqrt(2.0 / nodesIn);  // He-Initialisierung
            }
        }


        initializeGradients();
    }

    public void initializeGradients() {
        costGradientW = new double[nodesIn][nodesOut];
        costGradientB = new double[nodesOut];
    }


    public void applyGradients(double learningRate) {
        for (int out = 0; out < nodesOut; out++) {
            biases[out] -= costGradientB[out] * learningRate;

            for (int in = 0; in < nodesIn; in++) {
                weightsIn[in][out] -= costGradientW[in][out] * learningRate;
            }
        }
    }

    public void updateGradients(double[] nodeValues) {
        for (int nodeOut = 0; nodeOut < nodesOut; nodeOut++) {
            for (int nodeIn = 0; nodeIn < nodesIn; nodeIn++) {
                double derivativeCostWrtWeight = lastInputs[nodeIn] * nodeValues[nodeOut];

                costGradientW[nodeIn][nodeOut] += derivativeCostWrtWeight;
            }

            double derivativeCostWrtBias = 1 * nodeValues[nodeOut];
            costGradientB[nodeOut] += derivativeCostWrtBias;
        }
    }


    public double[] calculateOutputActivations(double[] inputs) {
        double[] activations = new double[nodesOut];
        double[] weightedInputs = new double[nodesOut];

        if (inputs.length != nodesIn) {
            System.out.println("Input-Doubles-Array not same length as required Input-Doubles-Array length for this layer!");
        }

        for (int out = 0; out < nodesOut; out++) {
            double weightedInput = 0;

            for (int in = 0; in < inputs.length; in++) {
                weightedInput += inputs[in] * weightsIn[in][out];
            }


            weightedInput += biases[out];
            weightedInputs[out] = weightedInput;
            activations[out] = activationFunction.activation(weightedInput);
        }

        lastActivations = activations;
        lastWeightedInputs = weightedInputs;
        lastInputs = inputs;

        return activations;
    }


    // TODO: Gehören nicht in Layer-Klasse sondern sind trainingsalgorithmusspezifisch
    public double[] calculateOutputLayerNodeValues(double[] expectedOutputs) {
        double[] nodeValues = new double[expectedOutputs.length];

        for (int i = 0; i < nodeValues.length; i++) {
            // Evaluate partial derivatives for current node: cost/activation & activation/weightedInput

            double costDerivative = costFunction.nodeCostDerivative(lastActivations[i], expectedOutputs[i]);
            double activationDerivative = activationFunction.derivative(lastWeightedInputs[i]);
            nodeValues[i] = activationDerivative * costDerivative;
        }

        return nodeValues;
    }

    public double[] calculateHiddenLayerNodeValues(Layer oldLayer, double[] oldNodeValues) {
        double[] newNodeValues = new double[nodesOut];

        for (int newNodeIndex = 0; newNodeIndex < newNodeValues.length; newNodeIndex++) {
            double newNodeValue = 0;
            for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.length; oldNodeIndex++) {
                double weightedInputDerivative = oldLayer.weightsIn[newNodeIndex][oldNodeIndex];
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }

            newNodeValue *= activationFunction.derivative(lastWeightedInputs[newNodeIndex]);
            newNodeValues[newNodeIndex] = newNodeValue;
        }

        return newNodeValues;
    }


    public int getNodesIn() {
        return nodesIn;
    }

    public int getNodesOut() {
        return nodesOut;
    }

    public double[][] getWeightsIn() {
        return weightsIn;
    }

    public double[] getBiases() {
        return biases;
    }

    public void setWeight(int in, int out, double newWeight) {
        weightsIn[in][out] = newWeight;
    }

    public void nudgeWeight(int in, int out, double deltaWeight) {
        weightsIn[in][out] += deltaWeight;
    }

    public void nudgeBias(int biasIndex, double deltaBias) {
        biases[biasIndex] += deltaBias;
    }

    public void setWeightCostGradient(int in, int out, double costGradient) {
        costGradientW[in][out] = costGradient;
    }

    public void setBiasCostGradient(int biasIndex, double costGradient) {
        costGradientB[biasIndex] = costGradient;
    }


}
