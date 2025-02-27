package de.jonathanebeling.neuralnetwork.cost_functions;

public class SumOfSquaredErrorsCost implements CostFunction {


    @Override
    public double cost(double outputActivation, double expectedActivation) {
        double error = outputActivation - expectedActivation;
        return error * error;
    }

    @Override
    public double derivative(double outputActivation, double expectedActivation) {
        return 2 * (outputActivation - expectedActivation);
    }
}
