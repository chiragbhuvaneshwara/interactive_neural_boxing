using System.Collections.Generic;
using UnityEngine;

namespace MultiMosiServer
{
    public partial class MultiMotionServer
    {
        [System.Serializable]
        public class punchInput
        {
            [SerializeField] public string hand;
            [SerializeField] public double[] target_left;
            [SerializeField] public double[] target_right;
            [SerializeField] public List<float> movement_dir;

        }

        public class ZeroPostureCommand
        {
            [SerializeField] public string name;
        }

    }

}


