using System.Collections.Generic;


namespace MultiMosiServer
{

    public partial class MultiMotionServer
    {
        public class Bone_x
        {
            public string Name { get; set; }
            public List<float> Position { get; set; }
            public double BoneRotation { get; set; }

            public Bone_x(string name, List<float> position, double rotation)
            {
                this.Name = name;
                this.Position = position;
                this.BoneRotation = rotation;
            }
        }

    }
}