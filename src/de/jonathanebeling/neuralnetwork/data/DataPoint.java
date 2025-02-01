package de.jonathanebeling.neuralnetwork.data;

import java.io.Serial;
import java.io.Serializable;

public class DataPoint implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    private double[] inputActivation;
    private final double[] expectedOutputActivation;

    public DataPoint(double[] inputActivation, double[] expectedOutputActivation) {
        this.inputActivation = inputActivation;
        this.expectedOutputActivation = expectedOutputActivation;
    }

    public DataPoint(double[] inputActivation, int expectedOutputLabel) {
        this.inputActivation = inputActivation;

        double[] expectedOutputActivation = new double[] {0,0,0,0,0,0,0,0,0,0};
        expectedOutputActivation[expectedOutputLabel] = 1;

        this.expectedOutputActivation = expectedOutputActivation;
    }

    public double[] getInputActivation() {
        return inputActivation;
    }

    public double[] getExpectedOutputActivation() {
        return expectedOutputActivation;
    }

    public void setInputActivation(double[] inputActivation) {this.inputActivation = inputActivation;}



}
