package de.jonathanebeling.neuralnetwork.network;

import de.jonathanebeling.neuralnetwork.activation_functions.ActivationFunction;
import de.jonathanebeling.neuralnetwork.cost_functions.CostFunction;
import de.jonathanebeling.neuralnetwork.utils.MathUtils;

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

        weightsIn = new double[nodesOut][nodesIn];
        biases = new double[nodesOut];


        //Random weights and biases on initialization
        for (int i = 0; i < nodesOut; i++) {
            for (int j = 0; j < nodesIn; j++) {
                weightsIn[i][j] = random.nextGaussian() * Math.sqrt(2.0 / nodesIn);  // He-Initialisierung
            }
        }
    }



    public synchronized void applyGradients(double learningRate, LayerTrainingInstance layerTrainingInstance) {
        for (int out = 0; out < nodesOut; out++) {
            biases[out] -= layerTrainingInstance.getCostGradientB()[out] * learningRate;

            for (int in = 0; in < nodesIn; in++) {
                weightsIn[out][in] -= layerTrainingInstance.getCostGradientW()[out][in] * learningRate;
            }
        }
    }


    public double[] calculateOutputActivations(double[] inputs) {
        double[] activations = new double[nodesOut];


        // Überprüfung der Inputs auf korrekte Länge
        if (inputs.length != nodesIn) {
            throw new IllegalArgumentException("Input-Doubles-Array not same length as required " +
                    "Input-Doubles-Array length for this layer!");
        }

        for (int out = 0; out < nodesOut; out++) {

            // Bilden des Skalar-Produkts
            double weightedInput = MathUtils.sumMultipliedArrays(inputs, weightsIn[out]);

            // Addieren von Bias-Wert
            weightedInput += biases[out];

            // Setzen des neuen Output-Werts jeder Node
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

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public CostFunction getCostFunction() {
        return costFunction;
    }
}
