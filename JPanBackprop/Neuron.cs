using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JPanBackprop
{
    public class Neuron
    {
        public double Bias;
        public double[] Weights;
        public double Output;
        public Func<double, double> Activation;
        public double PartialDerivative;
        public double BiasUpdate;
        public double PrevBiasUpdate;
        public double[] WeightUpdates;
        public double[] PrevWeightUpdates;

        public Neuron(Func<double, double> activation, int inputCount)
        {
            Activation = activation;
            Weights = new double[inputCount];
            WeightUpdates = new double[inputCount];
            PrevWeightUpdates = new double[inputCount];
        }

        public void Randomize(Random rand)
        {
            Bias = rand.NextDouble(-0.5, 0.5);
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = rand.NextDouble(-0.5, 0.5);
            }
        }

        public double Compute(double[] input)
        {
            Output = 0;
            for (int i = 0; i < input.Length; i++)
            {
                Output += input[i] * Weights[i];
            }
            
            Output = Activation(Output + Bias);

            return Output;
        }
    }
}
