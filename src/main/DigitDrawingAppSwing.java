package main;
import network.DataPoint;
import network.NeuralNetwork;
import utils.DisplayHelper;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.awt.image.ConvolveOp;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import javax.imageio.ImageIO;

public class DigitDrawingAppSwing extends JFrame {

    private BufferedImage canvasImage;
    private Graphics2D g2d;

    private DisplayHelper displayHelper = new DisplayHelper();
    private NeuralNetwork network = loadTrainedNetwork();



    public DigitDrawingAppSwing() {
        setTitle("Draw a Digit");
        setSize(560, 560);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Initialisiere das Canvas
        canvasImage = new BufferedImage(560, 560, BufferedImage.TYPE_INT_ARGB);
        g2d = canvasImage.createGraphics();
        g2d.setColor(Color.BLACK);
        g2d.fillRect(0, 0, canvasImage.getWidth(), canvasImage.getHeight());
        g2d.setColor(Color.WHITE);
        g2d.setStroke(new BasicStroke(20, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));

        JPanel drawPanel = getJPanel();

        JButton clearButton = new JButton("Clear");
        clearButton.addActionListener(e -> clearCanvas());

        JButton paint = new JButton("Paint");
        paint.addActionListener(e -> g2d.setColor(Color.WHITE));

        JButton erase = new JButton("Erase");
        erase.addActionListener(e -> g2d.setColor(Color.BLACK));

        JPanel buttonPanel = new JPanel();
        buttonPanel.add(clearButton);
        buttonPanel.add(paint);
        buttonPanel.add(erase);

        add(drawPanel, BorderLayout.CENTER);
        add(buttonPanel, BorderLayout.SOUTH);

        setVisible(true);

//        try (ExecutorService executor = Executors.newFixedThreadPool(1)) {
//            executor.submit(() -> startClassifyDrawing());
//        }
    }


    private JPanel getJPanel() {
        JPanel drawPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                g.drawImage(canvasImage, 0, 0, null);
            }
        };

        drawPanel.setPreferredSize(new Dimension(560, 560));
        drawPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                g2d.drawLine(e.getX(), e.getY(), e.getX(), e.getY());
                repaint();
            }
        });

        drawPanel.addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                g2d.drawLine(e.getX(), e.getY(), e.getX(), e.getY());
                repaint();
                classifyDrawing();
            }
        });
        return drawPanel;
    }

    private void clearCanvas() {
        g2d.setColor(Color.BLACK);
        g2d.fillRect(0, 0, canvasImage.getWidth(), canvasImage.getHeight());
        g2d.setColor(Color.WHITE);
        repaint();
    }

    private void classifyDrawing() {
        try {
            // Skaliere das Bild auf 28x28 und normalisiere es
            ImageIO.write(canvasImage, "png", new File("digit.png"));
//            BufferedImage blurredImage = applyGaussianBlur(canvasImage);
//            ImageIO.write(canvasImage, "png", new File("blurred_digit.png"));
            BufferedImage resizedImage = resizeImage(canvasImage, 28, 28);
            ImageIO.write(resizedImage, "png", new File("resized_digit.png"));
            BufferedImage blurredResizedImage = applyGaussianBlur(resizedImage);
            ImageIO.write(blurredResizedImage, "png", new File("blurred_resized_digit.png"));

            double[] inputData = imageToArray(blurredResizedImage);

            DataPoint dataPoint = new DataPoint(inputData, new double[10]);


//            displayHelper.printDigit(dataPoint);
            System.out.println("##############");
            displayHelper.printTop3Outputs(network.calculateOutputs(inputData));

            // Optional: Speichere das Bild für Debugging

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

//    private BufferedImage resizeImage(BufferedImage originalImage, int width, int height) {
//        BufferedImage resizedImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
//        Graphics2D g = resizedImage.createGraphics();
//        g.drawImage(originalImage, 0, 0, width, height, null);
//        g.dispose();
//        return resizedImage;
//    }

    private BufferedImage applyGaussianBlur(BufferedImage image) {
        float[] kernel = {
                1/8f, 1/4f, 1/8f,
                1/4f, 1f, 1/4f,
                1/8f, 1/4f, 1/8f
        };


        java.awt.image.ConvolveOp op = new java.awt.image.ConvolveOp(
                new java.awt.image.Kernel(3, 3, kernel),
                java.awt.image.ConvolveOp.EDGE_NO_OP,
                null
        );

        


        return op.filter(image, null);
    }

    private BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_BYTE_GRAY);
        java.awt.Graphics2D g2d = resizedImage.createGraphics();

        // Verwende bilineare Interpolation
        g2d.setRenderingHint(java.awt.RenderingHints.KEY_ANTIALIASING,
                java.awt.RenderingHints.VALUE_ANTIALIAS_ON);

        g2d.drawImage(originalImage, 0, 0, targetWidth, targetHeight, null);
        g2d.dispose();

        return resizedImage;
    }

    private double[] imageToArray(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        double[] inputData = new double[width * height];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

                int pixel = image.getRGB(x, y) & 0xFF; // Extrahiere Grauwert

                inputData[y * width + x] = (pixel/ 255.0); // Normalisiere auf [0, 1]
            }
        }

        return inputData;
    }



    private NeuralNetwork loadTrainedNetwork() {
        // Lade dein trainiertes neuronales Netzwerk
        try {
            return NeuralNetwork.load("src/saved_networks/relu-better-generalisation-3/epoch-44.ser");
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(DigitDrawingAppSwing::new);
    }

    private void startClassifyDrawing() {
        while (true){
            classifyDrawing();
            try {
                wait(200);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

    }
}