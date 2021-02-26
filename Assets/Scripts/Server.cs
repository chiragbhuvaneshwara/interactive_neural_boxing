using System.Collections;
using System.IO;
using UnityEngine;
using System.Net;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;

namespace MultiMosiServer
{

    public partial class MultiMotionServer
    {

        //public int port = 5000;
        //public string server = "127.0.0.1";
        string serverAddress = "http://127.0.0.1:5000/";

        private TPosture posture;
        public Traj trajectory;

        //private TVector3 direction = new TVector3(0, 0, 1);

        private bool active { get; set; } = true;
        private float total_anim_time = 0.0f;

        public float global_scale = 1.0f;

        //public string JsonPostWithResp(string route, dynamic VarToPost)
        public string JsonPostWithResp(string route, string VarToPost)
        {
            var httpWebRequest = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:5000/" + route);
            //Debug.Log("http://127.0.0.1:5000/" + route);
            httpWebRequest.ContentType = "application/json";
            httpWebRequest.Method = "POST";

            //string json = JsonUtility.ToJson(VarToPost);

            using (var streamWriter = new StreamWriter(httpWebRequest.GetRequestStream()))
            {
                //streamWriter.Write(json);
                streamWriter.Write(VarToPost);
            }

            var httpResponse = (HttpWebResponse)httpWebRequest.GetResponse();

            var streamReader = new StreamReader(httpResponse.GetResponseStream());
            string json_obj = streamReader.ReadToEnd();
            streamReader.Close();

            return json_obj;

        }

        public Transform GetPunchTargetTransform()
        {
            Transform punchTargetTransform;

            GameObject go = GameObject.Find("punch_target");
            punchTargetTransform = go.transform;
            //Debug.Log("punch target: " + punchTargetTransform.position);
            //Debug.Log(punchTargetTransform.position);
            return punchTargetTransform;

        }

        public string UpdatePunchTarget(string TargetHand)
        {
            punchInput punch_in = new punchInput();

            if (TargetHand == "right")
            {
                Vector3 l_hand_pos = GameObject.Find("LeftWrist_end").transform.position;
                //Debug.Log(l_hand_pos);
                var punchTargetTransform = GetPunchTargetTransform();
                Vector3 target = punchTargetTransform.position;


                punch_in.hand = "right";
                punch_in.target_left = new double[] { l_hand_pos.x, l_hand_pos.y, l_hand_pos.z };
                punch_in.target_right = new double[] { target.x, target.y, target.z };
            }
            else if (TargetHand == "left")
            {
                Vector3 r_hand_pos = GameObject.Find("RightWrist_end").transform.position;
                //Debug.Log(r_hand_pos);
                var punchTargetTransform = GetPunchTargetTransform();
                Vector3 target = punchTargetTransform.position;


                punch_in.hand = "left";
                punch_in.target_left = new double[] { target.x, target.y, target.z };
                punch_in.target_right = new double[] { r_hand_pos.x, r_hand_pos.y, r_hand_pos.z };
            }

            string punch_input = JsonConvert.SerializeObject(punch_in);
            string json_obj = JsonPostWithResp("punch_in", punch_input);
            
            //string data_1 = JObject.Parse(json_obj)["status"].ToString();
            BonePositions bp = JsonConvert.DeserializeObject<BonePositions>(json_obj);


            //Debug.Log("******************************");
            //Debug.Log(bp.pose);
            //Debug.Log(bp.message);
            //Debug.Log(bp.status);

            return json_obj;
        }


        public TPosture UpdatePunchTargetFetchPosture(string TargetHand)
        {
            punchInput punch_in = new punchInput();

            if (TargetHand == "right")
            {
                Vector3 l_hand_pos = GameObject.Find("LeftWrist_end").transform.position;
                //Debug.Log(l_hand_pos);
                var punchTargetTransform = GetPunchTargetTransform();
                Vector3 target = punchTargetTransform.position;


                punch_in.hand = "right";
                //punch_in.target_left = new double[] { l_hand_pos.x, l_hand_pos.y, l_hand_pos.z };
                punch_in.target_left = new double[] { 0, 0, 0};
                punch_in.target_right = new double[] { -target.x, target.y, target.z };
            }
            else if (TargetHand == "left")
            {
                Vector3 r_hand_pos = GameObject.Find("RightWrist_end").transform.position;
                //Debug.Log("r_hand_p:");
                //Debug.Log(r_hand_pos);
                var punchTargetTransform = GetPunchTargetTransform();
                Vector3 target = punchTargetTransform.position;


                punch_in.hand = "left";
                punch_in.target_left = new double[] { -target.x, target.y, target.z};
                punch_in.target_right = new double[] { 0, 0, 0};
            }
            else if (TargetHand == "none")
            {
                punch_in.hand = TargetHand;
                punch_in.target_left = new double[] {0, 0, 0};
                punch_in.target_right = new double[] {0, 0, 0};
            }

            string punch_input = JsonConvert.SerializeObject(punch_in);
            //Debug.Log(punch_input);
            string json_obj = JsonPostWithResp("fetch_frame", punch_input);

            TPosture posture = JsonConvert.DeserializeObject<TPosture>(json_obj);


            //Debug.Log("******************************");
            //Debug.Log(posture.Bones[0].name);
            ////Debug.Log(posture.Bone_map.Hips);
            //Debug.Log(posture.Bone_map["Hips"]);
            //Debug.Log(posture.Location.x);
            //Debug.Log(posture.Rotation);

            return posture;
        }


        public TPosture getZeroPosture()
        {
            ZeroPostureCommand zp_command = new ZeroPostureCommand();
            //zp_command.name = "fetch_zp";
            zp_command.name = "fetch_zp_reset";

            string zp_com_json = JsonConvert.SerializeObject(zp_command);
            string zp_json_obj = JsonPostWithResp("fetch_zp", zp_com_json);

            TPosture posture = JsonConvert.DeserializeObject<TPosture>(zp_json_obj);

            return posture;
        }

        //TODO: setup a route in the backend to register a new connection and assign the connection a unique ID.
        public void Start()
        {
            if (this.posture == null)
            {
                //TODO: setup a route to get zero posture
                this.posture = this.getZeroPosture();
            }


        }

        private TVector3 vec2Tvec(Vector3 x)
        {
            return new TVector3(-x.x, x.y, x.z);
        }
        private Vector3 Tvec2vec(TVector3 x)
        {
            return new Vector3(-(float)x.x, (float)x.y, (float)x.z);
        }

        public Vector3 GetJointPosition(string bone_name)
        {
            return Tvec2vec(this.posture.Bones[this.posture.Bone_map[bone_name]].position);
        }

        //TODO: setup a route
        public Vector3 GetGlobalLocation()
        {
            Vector3 pos = Tvec2vec(this.posture.Location) * this.global_scale;
            //pos.x = pos.x;
            return pos;
        }

        //TODO: setup a route
        public Quaternion GetGlobalRotation()
        {
            double rot = this.posture.Rotation;
            Quaternion q = Quaternion.AngleAxis(Mathf.Rad2Deg * ((float)rot), new Vector3(0, 1, 0));
            return q;
        }

        public Vector3 GetTrPos(string tr_name, int i)
        //public void GetArmTrPos(string tr_name, int i)
        {
            //Debug.Log(Tvec2vec(this.posture.arm_tr.rwt[i]).x);
            //Debug.Log(this.posture.arm_tr.rwt.Count);
            if (tr_name == "root")
            {
                return Tvec2vec(this.posture.arm_tr.rt[i]);
            }
            else if (tr_name == "root_vel")
            {
                return Tvec2vec(this.posture.arm_tr.rt_v[i]);
            }
            else if (tr_name == "right_wrist")
            {
                return Tvec2vec(this.posture.arm_tr.rwt[i]);
            }
            else if (tr_name == "left_wrist")
            {
                return Tvec2vec(this.posture.arm_tr.lwt[i]);
            }
            else if (tr_name == "right_wrist_vel")
            {
                return Tvec2vec(this.posture.arm_tr.rwt_v[i]);
            }
            else if (tr_name == "left_wrist_vel")
            {
                return Tvec2vec(this.posture.arm_tr.lwt_v[i]);
            }
            else
            {
                throw new InvalidOperationException("invalid trajectory name passed:" + tr_name);
            }
        }

        ////TODO: setup a route in the backend to terminate a connection with the specified unique ID.
        //public void stop()
        //{
        //    Debug.Log("Stoped motion server");
        //    this.client.Dispose();
        //}


        public void ManagedUpdate(string TargetHand) // void Update()
        {
            if (this.active)
            {
                //TODO: Setup root to get posture with bones, bone_map, root_global_loc, root_global_rot
                this.posture = this.UpdatePunchTargetFetchPosture(TargetHand);
            }
        }
    }

}