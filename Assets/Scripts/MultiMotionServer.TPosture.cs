using System.Collections.Generic;


namespace MultiMosiServer
{

    public partial class MultiMotionServer
    {
        //public class TPosture
        //{

        //    public List<TBone> Bones { get; set; }
        //    public Dictionary<string, int> Bone_map { get; set; }
        //    public List<float> Location { get; set; } //root position
        //    public double Rotation { get; set; }  //root rotation

        //    public TPosture(List<TBone> bones, Dictionary<string, int> bone_map, List<float> location, double rotation)
        //    {
        //        this.Bones = bones;
        //        this.Bone_map = bone_map;
        //        this.Location = location;
        //        this.Rotation = rotation;
        //    }
        //}

        // Root myDeserializedClass = JsonConvert.DeserializeObject<Root>(myJsonResponse); 
        public class Position
        {
            public double x { get; set; }
            public double y { get; set; }
            public double z { get; set; }
        }

        public class Rotation
        {
            public double w { get; set; }
            public double x { get; set; }
            public double y { get; set; }
            public double z { get; set; }
        }

        public class Bone
        {
            public string name { get; set; }
            public TVector3 position { get; set; }
            public Rotation rotation { get; set; }
            public List<string> children { get; set; }
            public string parent { get; set; }
        }

        //public class BoneMap
        //{
        //    public int Hips { get; set; }
        //    public int Chest { get; set; }
        //    public int Chest2 { get; set; }
        //    public int Chest3 { get; set; }
        //    public int Chest4 { get; set; }
        //    public int Neck { get; set; }
        //    public int Head { get; set; }
        //    public int RightCollar { get; set; }
        //    public int RightShoulder { get; set; }
        //    public int RightElbow { get; set; }
        //    public int RightWrist { get; set; }
        //    public int LeftCollar { get; set; }
        //    public int LeftShoulder { get; set; }
        //    public int LeftElbow { get; set; }
        //    public int LeftWrist { get; set; }
        //    public int RightHip { get; set; }
        //    public int RightKnee { get; set; }
        //    public int RightAnkle { get; set; }
        //    public int RightToe { get; set; }
        //    public int LeftHip { get; set; }
        //    public int LeftKnee { get; set; }
        //    public int LeftAnkle { get; set; }
        //    public int LeftToe { get; set; }
        //}

        public class TVector3
        {
            public double x { get; set; }
            public double y { get; set; }
            public double z { get; set; }

            public TVector3(double x, double y, double z)
            {
                this.x = -x;
                this.y = y;
                this.z = z;
            }
        }

        //public class Rwt
        //{
        //    public double x { get; set; }
        //    public double y { get; set; }
        //    public double z { get; set; }
        //}

        //public class Lwt
        //{
        //    public double x { get; set; }
        //    public double y { get; set; }
        //    public double z { get; set; }
        //}

        public class ArmTr
        {
            public List<TVector3> rwt { get; set; }
            public List<TVector3> lwt { get; set; }
        }


        public class TPosture
        {
            public List<Bone> Bones { get; set; }
            //public BoneMap Bone_map { get; set; }
            public Dictionary<string, int> Bone_map { get; set; }
            public TVector3 Location { get; set; }
            public double Rotation { get; set; }
            public ArmTr arm_tr { get; set; }

        }



    }
}
