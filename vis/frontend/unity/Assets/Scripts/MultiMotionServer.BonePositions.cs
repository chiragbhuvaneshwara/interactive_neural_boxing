using System.Collections.Generic;

namespace MultiMosiServer
{

    public partial class MultiMotionServer
    {
        public class BonePositions
        {
            public int status { get; set; }
            public string message { get; set; }
            public List<List<double>> pose { get; set; }
        }



















    }
}