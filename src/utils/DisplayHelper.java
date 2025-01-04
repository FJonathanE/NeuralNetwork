package utils;

import network.DataPoint;
import network.NeuralNetwork;

import java.util.*;

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

            if (getSortedIndices(output)[0] != getSortedIndices(dataPoints[i].getExpectedOutputActivation())[0]) {
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

            if (getSortedIndices(output)[0] != getSortedIndices(dataPoints[i].getExpectedOutputActivation())[0]) {
                wrongs++;
            }
        }

        System.out.println("Number of wrong identifications: " + wrongs + "/" + dataPoints.length);
    }

    public void printTop3Outputs(double[] output) {
        int[] sortedIndices = getSortedIndices(output);
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


    private int[] getSortedIndices(double[] array) {
        // Paare von Index und Wert erstellen
        Integer[] indices = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indices[i] = i;
        }

        // Sortiere Indizes basierend auf den Werten im Array
        Arrays.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer i1, Integer i2) {
                return Double.compare(array[i2], array[i1]); // Absteigend sortieren
            }
        });

        // Indizes als int[] zurückgeben
        return Arrays.stream(indices).mapToInt(Integer::intValue).toArray();
    }


}
