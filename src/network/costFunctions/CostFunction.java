package network.costFunctions;

import network.DataPoint;
import network.Layer;

import java.io.Serializable;

public interface CostFunction extends Serializable {


    double nodeCost(double outputActivation, double expectedActivation);
    double nodeCostDerivative(double outputActivation, double expectedActivation);

    default double dataPointCost(DataPoint dataPoint, double[] outputs){

        double cost = 0;

        for (int nodeOut = 0; nodeOut < outputs.length; nodeOut++) {
            cost += nodeCost(outputs[nodeOut], dataPoint.getExpectedOutputActivation()[nodeOut]);
        }

        return cost;
    }
}
