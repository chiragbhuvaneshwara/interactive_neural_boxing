using UnityEngine;

public partial class Player
{
    struct Bone
    {
        public Quaternion rotation;
        public Vector3 direction;

        public Bone(Quaternion rotation, Vector3 direction)
        {
            this.rotation = rotation;
            this.direction = direction;
        }
    }

}
