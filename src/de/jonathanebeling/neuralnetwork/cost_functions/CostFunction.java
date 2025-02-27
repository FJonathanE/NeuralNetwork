package de.jonathanebeling.neuralnetwork.cost_functions;

import de.jonathanebeling.neuralnetwork.data.DataPoint;

import java.io.Serializable;

public interface CostFunction extends Serializable {


    double cost(double outputActivation, double expectedActivation);
    double derivative(double outputActivation, double expectedActivation);

    default double dataPointCost(DataPoint dataPoint, double[] outputs){

        double cost = 0;

        for (int nodeOut = 0; nodeOut < outputs.length; nodeOut++) {
            cost += cost(outputs[nodeOut], dataPoint.getExpectedOutputActivation()[nodeOut]);
        }

        return cost;
    }
}
