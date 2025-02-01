package de.jonathanebeling.neuralnetwork.network;

import de.jonathanebeling.neuralnetwork.activationFunctions.ActivationFunction;
import de.jonathanebeling.neuralnetwork.costFunctions.CostFunction;

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
    }



    public synchronized void applyGradients(double learningRate, LayerTrainingInstance layerTrainingInstance) {
        for (int out = 0; out < nodesOut; out++) {
            biases[out] -= layerTrainingInstance.getCostGradientB()[out] * learningRate;

            for (int in = 0; in < nodesIn; in++) {
                weightsIn[in][out] -= layerTrainingInstance.getCostGradientW()[in][out] * learningRate;
            }
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


        return activations;
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

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public CostFunction getCostFunction() {
        return costFunction;
    }
}
