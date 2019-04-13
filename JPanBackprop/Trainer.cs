using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JPanBackprop
{
    public class Trainer
    {
        Network network;
        public Trainer(Network net)
        {
            network = net;
        }

        public void GradientDescent(double[][] inputs, double[][] desiredOutput)
        {
            ClearUpdates();

            for (int i = 0; i < inputs.Length; i++)
            {
                network.Compute(inputs[i]);
                CalculateError(desiredOutput[i]);
                CalculateUpdates(inputs[i], 0.5);
            }

            
        }

        public void ClearUpdates()
        {
            foreach (Layer layer in network.Layers)
            {
                foreach (Neuron neuron in layer.Neurons)
                {
                    neuron.WeightUpdate = 0;
                    neuron.BiasUpdate = 0;
                }
            }
        }

        public void CalculateError(double[] desiredOutput)
        {
            Layer outputLayer = network.Layers[network.Layers.Length - 1];
            for (int i = 0; i < outputLayer.Neurons.Length; i++)
            {
                Neuron neuron = outputLayer.Neurons[i];
                double error = neuron.Output - desiredOutput[i];
                neuron.PartialDerivative = error * (neuron.Output * (1 - neuron.Output));
            }

            for (int i = outputLayer.Neurons.Length - 2; i >= 0; i--)
            {
                Layer currLayer = network.Layers[i];
                Layer nextLayer = network.Layers[i + 1];
                
                for (int j = 0; j < currLayer.Neurons.Length; j++)
                {
                    Neuron neuron = currLayer.Neurons[j];
                    double error = 0.0;

                    foreach (Neuron nextNeuron in nextLayer.Neurons)
                    {
                        error += nextNeuron.PartialDerivative * nextNeuron.Weights[j];
                    }

                    neuron.PartialDerivative = error * (neuron.Output * (1 - neuron.Output));
                }
            }
        }

        public void CalculateUpdates(double[] input, double learningRate)
        {
            Layer inputLayer = network.Layers[0];
            foreach (Neuron neuron in inputLayer.Neurons)
            {
                for (int i = 0; i < input.Length; i++)
                {
                    neuron.WeightUpdate = learningRate * neuron.PartialDerivative * input[i];
                }
                neuron.BiasUpdate = learningRate * neuron.PartialDerivative;
            }
            
            for (int i = 1; i < network.Layers.Length; i++)
            {
                Layer currLayer = network.Layers[i];
                Layer prevLayer = network.Layers[i - 1];

                foreach (Neuron neuron in currLayer.Neurons)
                {
                    for (int j = 0; j < prevLayer.Neurons.Length; j++)
                    {
                        neuron.WeightUpdate = learningRate * neuron.PartialDerivative * prevLayer.Neurons[j].Output;
                    }
                    neuron.BiasUpdate = learningRate * neuron.PartialDerivative;
                }
            }
        }

        public void ApplyUpdates()
        {
            foreach (Layer layer in network.Layers)
            {
                foreach (Neuron neuron in layer.Neurons)
                {
                    // add momentum stuff
                    //wC = WU + (PWU * M)
                    double weightChange = neuron.WeightUpdate;
                }
            }
        }
    }
}
