package de.jonathanebeling.neuralnetwork.activationFunctions;

public class SigmoidActivation implements ActivationFunction {


    @Override
    public double activation(double weightedInput) {
        return 1 / (1 + Math.exp(-weightedInput));
    }

    @Override
    public double derivative(double weightedInput) {
        double activation = activation(weightedInput);
        return activation * (1 - activation);
    }
}
