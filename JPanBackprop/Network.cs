using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JPanBackprop
{
    public class Network
    {
        public Layer[] Layers;
        public double[] Output => Layers[Layers.Length - 1].Output;

        public Network(Func<double, double> activation, int inputCount, params int[] neuronsPerLayer)
        {
            Layers = new Layer[neuronsPerLayer.Length];

            Layers[0] = new Layer(activation, inputCount, neuronsPerLayer[0]);
            for (int i = 1; i < neuronsPerLayer.Length; i++)
            {
                Layers[i] = new Layer(activation, neuronsPerLayer[i - 1], neuronsPerLayer[i]);
            }
        }

        public void Randomize(Random rand)
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].Randomize(rand);
            }
        }

        public double[] Compute(double[] input)
        {
            double[] output = input;
            for (int i = 0; i < Layers.Length; i++)
            {
                output = Layers[i].Compute(output);
            }
            return output;
        }

        public void Mutate(Random rand, double rate)
        {
            foreach (Layer layer in Layers)
            {
                foreach (Neuron neuron in layer.Neurons)
                {
                    if (rand.NextDouble() < rate)
                    {
                        neuron.Bias = neuron.Bias * rand.NextDouble(0.1, 2) * rand.RandomSign();
                    }

                    for (int w = 0; w < neuron.Weights.Length; w++)
                    {
                        if (rand.NextDouble() < rate)
                        {
                            neuron.Weights[w] = neuron.Weights[w] * rand.NextDouble(0.1, 2) * rand.RandomSign();
                        }
                    }
                }
            }
        }

        public double MAE(double[][] input, double[][] desiredOutput)
        {
            double mae = 0;
            for (int r = 0; r < input.Length; r++)
            {
                Compute(input[r]);
                double rowError = 0;
                for (int i = 0; i < desiredOutput[r].Length; i++)
                {
                    rowError += Math.Abs(Output[i] - desiredOutput[r][i]);
                }
                rowError /= desiredOutput[r].Length;
                mae += rowError;
            }
            return mae / input.Length;
        }
    }
}
