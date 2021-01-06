using System.Collections.Generic;


namespace MultiMosiServer
{

    public partial class MultiMotionServer
    {
        public class TPosture
        {

            public List<TBone> Bones { get; set; }
            public Dictionary<string, int> Bone_map { get; set; }
            public List<float> Location { get; set; }
            public double Rotation { get; set; }

            public TPosture(List<TBone> bones, Dictionary<string, int> bone_map, List<float> location, double rotation)
            {
                this.Bones = bones;
                this.Bone_map = bone_map;
                this.Location = location;
                this.Rotation = rotation;
            }
        }

    }
}