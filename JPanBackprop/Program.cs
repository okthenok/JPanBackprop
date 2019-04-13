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

            
        }
    }
}
