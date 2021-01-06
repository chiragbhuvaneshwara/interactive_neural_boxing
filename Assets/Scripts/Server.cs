//using System.Collections;
//using System.Collections.Generic;
//using UnityEngine;


//namespace MultiMosiServer
//{

//    public partial class MultiMotionServer : MonoBehaviour
//    {

//        public int port = 5000;
//        public string server = "127.0.0.1";

//        private TPosture posture;

//        //private TVector3 direction = new TVector3(0, 0, 1);

//        private bool active { get; set; } = true;
//        private float total_anim_time = 0.0f;

//        public float global_scale = 1.0f;


//        //TODO: setup a route in the backend to register a new connection and assign the connection a unique ID.
//        void Start()
//        {
//            string serverAddress = "http://" + server + ":" + port + "/";


//        }

//        public Vector3 GetJointPosition(string bone_name)
//        {
//            return Tvec2vec(this.posture.Bones[this.posture.Bone_map[bone_name]].Position);
//        }

//        public TPosture GetZeroPosture()
//        {
//            return this.client.getZeroPosture();
//        }

//        //TODO: setup a route
//        public Vector3 GetGlobalLocation()
//        {
//            Vector3 pos = Tvec2vec(this.posture.Location) * this.global_scale;
//            pos.x = pos.x;
//            return pos;
//        }

//        //TODO: setup a route
//        public Quaternion GetGlobalRotation()
//        {
//            double rot = this.posture.Rotation;
//            Quaternion q = Quaternion.AngleAxis(Mathf.Rad2Deg * ((float)rot), new Vector3(0, 1, 0));
//            return q;
//        }


//        //TODO: setup a route in the backend to terminate a connection with the specified unique ID.
//        public void stop()
//        {
//            Debug.Log("Stoped motion server");
//            this.client.Dispose();
//        }


//        private Vector3 List2Vector3(List<float> values)
//        {
//            return new Vector3(float.Parse(values[0]), float.Parse(values[1]), float.Parse(values[2]));
//        }

//        private TVector3 vec2Tvec(Vector3 x)
//        {
//            return new TVector3(-x.x, x.y, x.z);
//        }
//        private Vector3 Tvec2vec(TVector3 x)
//        {
//            return new Vector3(-(float)x.X, (float)x.Y, (float)x.Z);
//        }



//        public Vector3 getDirection()
//        {
//            return Tvec2vec(this.direction);
//        }

//        public void setDirection(Vector3 direction)
//        {
//            this.direction = vec2Tvec(direction);
//            if (direction.magnitude > 0.1)
//            {
//                this.gait = TGait.walking;
//            }
//            else
//            {
//                this.gait = TGait.standing;
//            }
//        }

//        public void Running()
//        {
//            this.gait = TGait.running;
//        }

//        public void Walking()
//        {
//            this.gait = TGait.walking;
//        }

//        // Update is called once per frame
//        public void ManagedUpdate() // void Update()
//        {
//            if (this.active)
//            {

//                if (id < 0)
//                {
//                    this.id = this.client.registerSession();
//                }
//                // total_anim_time += Time.deltaTime;
//                // this.direction = new TVector3(2 * Mathf.Cos(total_anim_time % (2 * Mathf.PI)), 0, 2 * Mathf.Sin(total_anim_time % (2 * Mathf.PI)));

//                if (this.posture == null)
//                {
//                    //TODO: setup a route to get zero posture
//                    this.posture = this.client.getZeroPosture();
//                }

//                //TODO: 
//                this.posture = this.client.fetchFrame(Time.deltaTime, this.posture, this.direction, this.gait, id);
//            }
//        }


//    }
//}