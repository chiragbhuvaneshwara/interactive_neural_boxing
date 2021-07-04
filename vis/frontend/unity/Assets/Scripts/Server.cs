using System.Collections;
using System.Collections.Generic;
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

        public string GetZpJson(string route)
        {
            HttpWebRequest request = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:5000/" + route);
            //request.AutomaticDecompression = DecompressionMethods.GZip | DecompressionMethods.Deflate;

            var httpResponse = (HttpWebResponse)request.GetResponse();

            var streamReader = new StreamReader(httpResponse.GetResponseStream());
            string json_obj = streamReader.ReadToEnd();
            streamReader.Close();

            return json_obj;
        }


        public string JsonPostWithResp(string route, string VarToPost)
        {
            var httpWebRequest = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:5000/" + route);
            //Debug.Log("http://127.0.0.1:5000/" + route);
            httpWebRequest.ContentType = "application/json";
            httpWebRequest.Method = "POST";

            // TODO: Set timeout correctly when not debugging
            httpWebRequest.Timeout = System.Threading.Timeout.Infinite;

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


        public TPosture UpdatePunchTargetFetchPosture(string TargetHand, List<float> MovementDir, int TrajectoryPtsReached)
        {
            punchInput punch_in = new punchInput();

            if (TargetHand == "right")
            {
                Vector3 l_hand_pos = GameObject.Find("LeftWrist_end").transform.position;
                var punchTargetTransform = GetPunchTargetTransform();
                Vector3 target = punchTargetTransform.position;


                punch_in.hand = "right";
                punch_in.target_reached = TrajectoryPtsReached;
                punch_in.target_left = new double[] { 0, 0, 0};
                punch_in.target_right = new double[] { target.x, target.y, target.z };
            }
            else if (TargetHand == "left")
            {
                Vector3 r_hand_pos = GameObject.Find("RightWrist_end").transform.position;
                var punchTargetTransform = GetPunchTargetTransform();
                Vector3 target = punchTargetTransform.position;


                punch_in.hand = "left";
                punch_in.target_reached = TrajectoryPtsReached;
                punch_in.target_left = new double[] { target.x, target.y, target.z};
                punch_in.target_right = new double[] { 0, 0, 0};
            }
            else if (TargetHand == "none")
            {
                punch_in.hand = TargetHand;
                punch_in.target_left = new double[] {0, 0, 0};
                punch_in.target_right = new double[] {0, 0, 0};
            }

            punch_in.movement_dir = MovementDir;

            string punch_input = JsonConvert.SerializeObject(punch_in);
            string json_obj = JsonPostWithResp("fetch_frame", punch_input);

            TPosture posture = JsonConvert.DeserializeObject<TPosture>(json_obj);

            return posture;
        }


        public TPosture getZeroPosture()
        {
            //ZeroPostureCommand zp_command = new ZeroPostureCommand();
            //zp_command.name = "fetch_zp";
            //zp_command.name = "fetch_zp_reset";

            //string zp_com_json = JsonConvert.SerializeObject(zp_command);
            //string zp_json_obj = JsonPostWithResp("fetch_zp", zp_com_json);
            string zp_json_obj = GetZpJson("fetch_zp");

            TPosture posture = JsonConvert.DeserializeObject<TPosture>(zp_json_obj);

            return posture;
        }

        public void Start()
        {
            if (this.posture == null)
            {
                this.posture = this.getZeroPosture();
            }


        }

        private TVector3 vec2Tvec(Vector3 x)
        {
            return new TVector3(x.x, x.y, x.z);
        }
        private Vector3 Tvec2vec(TVector3 x)
        {
            return new Vector3((float)x.x, (float)x.y, (float)x.z);
        }

        public Vector3 GetJointPosition(string bone_name)
        {
            return Tvec2vec(this.posture.Bones[this.posture.Bone_map[bone_name]].position);
        }

        public Vector3 GetGlobalLocation()
        {
            Vector3 pos = Tvec2vec(this.posture.Location) * this.global_scale;
            //pos.x = pos.x;
            return pos;
        }

        public Quaternion GetGlobalRotation()
        {
            double rot = this.posture.Rotation;
            Quaternion q = Quaternion.AngleAxis(Mathf.Rad2Deg * ((float)rot), new Vector3(0, 1, 0));
            return q;
        }

        public Vector3 GetTrPos(string tr_name, int i)
        //public void GetArmTrPos(string tr_name, int i)
        {
            //Debug.Log(Tvec2vec(this.posture.traj.rwt[i]).x);
            //Debug.Log(this.posture.traj.rwt.Count);
            if (tr_name == "root")
            {
                return Tvec2vec(this.posture.traj.rt[i]);
            }
            else if (tr_name == "root_vel")
            {
                return Tvec2vec(this.posture.traj.rt_v[i]);
            }
            else if (tr_name == "right_wrist")
            {
                return Tvec2vec(this.posture.traj.rwt[i]);
            }
            else if (tr_name == "left_wrist")
            {
                return Tvec2vec(this.posture.traj.lwt[i]);
            }
            else if (tr_name == "right_wrist_vel")
            {
                return Tvec2vec(this.posture.traj.rwt_v[i]);
            }
            else if (tr_name == "left_wrist_vel")
            {
                return Tvec2vec(this.posture.traj.lwt_v[i]);
            }
            else
            {
                throw new InvalidOperationException("invalid trajectory name passed:" + tr_name);
            }
        }

        //public void stop()
        //{
        //    Debug.Log("Stoped motion server");
        //    this.client.Dispose();
        //}

        public void ManagedUpdate(string TargetHand, List<float> MovementDir, int TrajPtsReached) // void Update()
        {
            if (this.active)
            {
                this.posture = this.UpdatePunchTargetFetchPosture(TargetHand, MovementDir, TrajPtsReached);
            }
        }
    }

}