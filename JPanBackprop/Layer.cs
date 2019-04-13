using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JPanBackprop
{
    public class Layer
    {
        public Neuron[] Neurons;
        public double[] Output;

        public Layer(Func<double, double> activation, int inputCount, int neuronCount)
        {
            Output = new double[neuronCount];
            Neurons = new Neuron[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                Neurons[i] = new Neuron(activation, inputCount);
            }
        }

        public void Randomize(Random rand)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].Randomize(rand);
            }
        }
        
        public double[] Compute(double[] input)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Output[i] = Neurons[i].Compute(input);
            }
            return Output;
        }
    }
}
