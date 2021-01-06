using System.Collections.Generic;


namespace MultiMosiServer
{

    public partial class MultiMotionServer
    {
        public class TBone
        {
            public string Name { get; set; }
            public List<float> Position { get; set; }
            public double BoneRotation { get; set; }

            public TBone(string name, List<float> position, double rotation)
            {
                this.Name = name;
                this.Position = position;
                this.BoneRotation = rotation;
            }
        }

    }
}