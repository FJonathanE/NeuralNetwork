package de.jonathanebeling.neuralnetwork.activationFunctions;

import java.io.Serializable;

public interface ActivationFunction extends Serializable {

    double activation(double weightedInput);
    double derivative(double weightedInput);
}
