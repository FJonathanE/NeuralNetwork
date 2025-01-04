package network.activationFunctions;

public class ReLuActivation implements ActivationFunction{
    @Override
    public double activation(double weightedInput) {
        return weightedInput > 0 ? weightedInput : 0.01 * weightedInput;
    }

    @Override
    public double derivative(double weightedInput) {
        return weightedInput > 0 ? 1 : 0.01;
    }
}
