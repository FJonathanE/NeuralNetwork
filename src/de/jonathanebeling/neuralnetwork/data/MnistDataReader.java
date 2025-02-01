package de.jonathanebeling.neuralnetwork.data;

import java.io.*;

public class MnistDataReader  {

    public DataPoint[] readData(String dataFilePath, String labelFilePath) throws IOException {

        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)));
        int magicNumber = dataInputStream.readInt();
        int numberOfItems = dataInputStream.readInt();
        int nRows = dataInputStream.readInt();
        int nCols = dataInputStream.readInt();

        System.out.println("Amount of items found: " + numberOfItems);

        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));
        int labelMagicNumber = labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();

        System.out.println("Amount of labels found: " + numberOfLabels);
        System.out.println(" ");


        DataPoint[] data = new DataPoint[numberOfItems];

        assert numberOfItems == numberOfLabels;

        for(int i = 0; i < numberOfItems; i++) {

            double[] inputActivation = new double[nRows*nCols];
            for (int r = 0; r < nRows; r++) {
                for (int c = 0; c < nCols; c++) {
                    inputActivation[r*nCols + c] = ((double) dataInputStream.readUnsignedByte() /255);
                }
            }
            data[i] = new DataPoint(inputActivation, labelInputStream.readUnsignedByte());
        }
        dataInputStream.close();
        labelInputStream.close();
        return data;
    }
}