using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JPanBackprop
{
    class Program
    {
        static void Main(string[] args)
        {
            Network network = new Network(a => 1 / (1 + Math.Exp(-a)), 2, 2, 1);
            Trainer trainer = new Trainer(network);
            double[][] inputs = new double[][]
            {
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 },
                new double[] { 0, 0 }
            };
            double[][] outputs = new double[][]
            {
                new double[] {1},
                new double[] {1},
                new double[] {0},
                new double[] {0}
            };

            while (network.MAE(inputs, outputs) > 0.1)
            {
                trainer.GradientDescent(inputs, outputs);

                for (int i = 0; i < outputs.Length; i++)
                {
                    double output = network.Compute(inputs[i])[0];
                    Console.WriteLine(Math.Round(output));
                }
                Console.WriteLine();
            }
        }
    }
}
