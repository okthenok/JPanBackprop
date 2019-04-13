using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JPanBackprop
{
    public static class Extensions
    {
        public static double NextDouble(this Random rand, double min, double max)
        {
            return rand.NextDouble() * (max - min) + min;
        }

        public static int RandomSign(this Random source)
        {
            return source.Next(2) * 2 - 1;
        }
    }
}
