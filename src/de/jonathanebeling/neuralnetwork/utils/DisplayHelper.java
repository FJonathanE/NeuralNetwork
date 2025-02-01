package de.jonathanebeling.neuralnetwork.utils;

import de.jonathanebeling.neuralnetwork.data.DataPoint;
import de.jonathanebeling.neuralnetwork.network.NeuralNetwork;

public class DisplayHelper {


    public void printDigit(DataPoint dataPoint) {
        for (int r = 0; r < 28; r++) {
            String row = "";
            for (int c = 0; c < 28; c++) {
                row = row + getGradient(dataPoint.getInputActivation()[r * 28 + c]) + "  ";
            }
            System.out.println(row);
        }
    }


    public void printWrongDigits(NeuralNetwork network, DataPoint[] dataPoints) {
        int wrongs = 0;

        for (int i = 0; i < dataPoints.length; i++) {
            double[] output = network.calculateOutputs(dataPoints[i].getInputActivation());

            if (MathUtils.getSortedIndices(output)[0] != MathUtils.getSortedIndices(dataPoints[i].getExpectedOutputActivation())[0]) {
                wrongs++;

                System.out.println("============================");
                printDigit(dataPoints[i]);
                System.out.println("Expected Output: " + expextedOutputActivationToIndex(dataPoints[i].getExpectedOutputActivation()));
                System.out.println("Output by Network: ");
                printTop3Outputs(output);
                System.out.println("============================");

            }
        }

        System.out.println("Number of wrong identifications: " + wrongs + "/" + dataPoints.length);
    }
    public void printNumberOfWrongs(NeuralNetwork network, DataPoint[] dataPoints) {
        int wrongs = 0;

        for (int i = 0; i < dataPoints.length; i++) {
            double[] output = network.calculateOutputs(dataPoints[i].getInputActivation());

            if (MathUtils.getSortedIndices(output)[0] != MathUtils.getSortedIndices(dataPoints[i].getExpectedOutputActivation())[0]) {
                wrongs++;
            }
        }

        System.out.println("Number of wrong identifications: " + wrongs + "/" + dataPoints.length);
        System.out.println("Correct percentage: " + MathUtils.roundDecimalPoints((100-((double) wrongs /dataPoints.length)*100), 3) + "%");
    }

    public void printTop3Outputs(double[] output) {
        int[] sortedIndices = MathUtils.getSortedIndices(output);
        for (int i = 0; i < 3; i++) {
            System.out.println(sortedIndices[i] + ": " + Math.round(output[sortedIndices[i]] * 100.0) / 100.0);
        }
    }

    private char getGradient(double percentage) {
        char[] gradient = {' ', '·', '.', '-', '=', '#', '░', '▒', '▓', '█'};

        int index = (int) (percentage * (gradient.length - 1));

        return gradient[index];
    }

    private int expextedOutputActivationToIndex(double[] expectedOutputActivation) {
        for (int i = 0; i < expectedOutputActivation.length; i++) {
            if (expectedOutputActivation[i] == 1) return i;
        }
        return -1;
    }





}
