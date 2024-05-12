// Machine Learning
// File name: Program.cs
// Code It Yourself with .NET, 2024

// MNIST - Modified National Institute of Standards and Technology database

using MachineLearning;
using MachineLearning.NeuralNetwork;
using MachineLearning.NeuralNetwork.DataSources;
using MachineLearning.NeuralNetwork.Layers;
using MachineLearning.NeuralNetwork.Losses;
using MachineLearning.NeuralNetwork.Operations;
using MachineLearning.NeuralNetwork.Optimizers;
using MachineLearning.NeuralNetwork.ParamInitializers;

namespace MnistTests;

internal class Program
{
    private static void Main(string[] args)
    {
        Matrix train = Matrix.LoadCsv(".\\data\\mnist_train_small.csv");
        Matrix test = Matrix.LoadCsv(".\\data\\mnist_test.csv");

        (Matrix xTrain, Matrix yTrain) = Split(train);
        (Matrix xTest, Matrix yTest) = Split(test);

        // Scale xTrain and xTest to mean 0, variance 1
        Console.WriteLine("Scale data to mean 0...");

        float mean = xTrain.Mean();
        xTrain.AddInPlace(-mean);
        xTest.AddInPlace(-mean);

        Console.WriteLine($"xTrain min: {xTrain.Min()}");
        Console.WriteLine($"xTest min: {xTest.Min()}");
        Console.WriteLine($"xTrain max: {xTrain.Max()}");
        Console.WriteLine($"xTest max: {xTest.Max()}");

        Console.WriteLine("\nScale data to variance 1...");

        float std = xTrain.Std();
        xTrain.DivideInPlace(std);
        xTest.DivideInPlace(std);

        Console.WriteLine($"xTrain min: {xTrain.Min()}");
        Console.WriteLine($"xTest min: {xTest.Min()}");
        Console.WriteLine($"xTrain max: {xTrain.Max()}");
        Console.WriteLine($"xTest max: {xTest.Max()}");

        SimpleDataSource dataSource = new(xTrain, yTrain, xTest, yTest);

        // Define the network.
        NeuralNetwork model = new(
            layers: [
                new DenseLayer(89, new Tanh(), new RandomInitializer(12345)),
                // TODO: Try to change Linear to Softmax, and SoftmaxCrossEntropyLoss to CrossEntropyLoss.
                new DenseLayer(10, new Linear(), new RandomInitializer(12345))
            ],
            lossFunction: new SoftmaxCrossEntropyLoss()
        );

        Console.WriteLine("\nStart training...\n");

        Trainer trainer = new(model, new StochasticGradientDescent(0.1f));
        trainer.Fit(dataSource, EvalFunction, epochs: 10, evalEveryEpochs: 1, batchSize: 100);

        Console.ReadLine();
    }

    private static float EvalFunction(NeuralNetwork neuralNetwork, Matrix xEvalTest, Matrix yEvalTest)
    {
        Matrix prediction = neuralNetwork.Forward(xEvalTest);
        Matrix predictionArgmax = prediction.Argmax();

        int rows = predictionArgmax.Array.GetLength(0);
        if (rows != yEvalTest.GetDimension(Dimension.Rows))
        {
            throw new ArgumentException("Number of samples in prediction and yEvalTest do not match.");
        }

        int hits = 0;
        for (int row = 0; row < rows; row++)
        {
            int predictedDigit = Convert.ToInt32(predictionArgmax.Array.GetValue(row, 0));
            if ((float)yEvalTest.Array.GetValue(row, predictedDigit)! == 1f)
                hits++;
        }

        float accuracy = (float)hits / rows;
        return accuracy;
    }

    private static (Matrix xTest, Matrix yTest) Split(Matrix source)
    {
        // Split into xTest (all columns except the first one) and yTest (a one-hot table from the first column with values from 0 to 9).

        Matrix xTest = source.GetColumns(1..source.GetDimension(Dimension.Columns));
        Matrix yTest = source.GetColumn(0);

        // Convert yTest to a one-hot table.
        Matrix oneHot = new(yTest.GetDimension(Dimension.Rows), 10);
        for (int row = 0; row < yTest.GetDimension(Dimension.Rows); row++)
        {
            int value = Convert.ToInt32(yTest.Array.GetValue(row, 0));
            oneHot.Array.SetValue(1, row, value);
        }

        return (xTest, oneHot);
    }
}
