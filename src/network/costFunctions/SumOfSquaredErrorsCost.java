package network.costFunctions;

import network.DataPoint;

public class SumOfSquaredErrorsCost implements CostFunction {


    @Override
    public double nodeCost(double outputActivation, double expectedActivation) {
        double error = outputActivation - expectedActivation;
        return error * error;
    }

    @Override
    public double nodeCostDerivative(double outputActivation, double expectedActivation) {
        return 2 * (outputActivation - expectedActivation);
    }
}
