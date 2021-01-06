using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using UnityEngine;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

public partial class Player : MonoBehaviour
{
    private bool leftMousePressed;
    private bool rightMousePressed;
    private bool midMousePressed;

    public Transform rootBone = null;
    public Transform characterTransform = null;

    private Dictionary<string, Bone> bonemap = new Dictionary<string, Bone>();

    public bool start = false;
    public bool fix_global_pos = false;
    public float velocity_scale = 1.0f;

    private Vector3 global_offset = Vector3.zero;

    //public MultiMosiServer.MultiMotionServer server = null;

    //private void initializeBones(Transform t)
    //{
    //    //TODO: update for Xsens Awinda skeleton
    //    if (t.name.Contains("end"))
    //    {
    //        // End condition: end joint of final bone reached
    //        return;
    //    }

    //    // apply transforms
    //    Quaternion startInverseRotation = new Quaternion(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w);
    //    int childID = 0;
    //    if (t.childCount > 1)
    //    {
    //        childID = 1;
    //    }
    //    Vector3 defaultDirection = new Vector3(0, 1, 0);
    //    if (t.childCount > 0)
    //    {
    //        Vector3 theChildPosition = t.GetChild(childID).transform.position;
    //        Vector3 thePosition = t.position;
    //        defaultDirection = (theChildPosition - thePosition);
    //    }

    //    Bone b = new Bone(startInverseRotation, defaultDirection);

    //    this.bonemap.Add(t.name, b);

    //    // Iteration step
    //    for (int i = 0; i < t.childCount; i++)
    //    {
    //        initializeBones(t.GetChild(i));
    //    }
    //}

    //// Start is called before the first frame update
    //void Start()
    //{
    //    //RigidBodyComp = GetComponent<Rigidbody>();
    //    this.global_offset += this.transform.position;
    //    this.initializeBones(this.rootBone);
    //}

    //private void processTransforms(Transform t)
    //{
    //    //TODO: update for Xsens Awinda skeleton
    //    if (t.name.Contains("end"))
    //    {
    //        // End condition: end joint of final bone reached
    //        return;
    //    }

    //    // apply transforms
    //    Vector3 pos = this.server.GetJointPosition(t.name);// * 1.7f;  //TODO: update for your code
    //    t.position = pos;

    //    if (t.childCount > 0 && !t.GetChild(0).name.Contains("end"))
    //    {
    //        int childID = 0;
    //        if (t.childCount > 1)
    //        {
    //            childID = 1;
    //        }
    //        Vector3 toChildPos = this.server.GetJointPosition(t.GetChild(childID).name) - t.position; //* 1.7f //TODO: update for your code
    //        t.rotation = Quaternion.FromToRotation(this.bonemap[t.name].direction, toChildPos) * this.bonemap[t.name].rotation;


    //    }


    //    // Iteration step
    //    for (int i = 0; i < t.childCount; i++)
    //    {
    //        processTransforms(t.GetChild(i));
    //    }
    //}

    //void LateUpdate()
    //{
    //    //TODO: update processTransforms to obtain relevant joint positions from your code

    //    if (start)
    //    {
    //        this.rootBone.rotation = Quaternion.identity;
    //        processTransforms(this.rootBone);

    //        if (!this.fix_global_pos)
    //        {
    //            Vector3 q = this.server.GetGlobalRotation().eulerAngles;
    //            Vector3 qt = this.rootBone.rotation.eulerAngles;
    //            qt.y = -q.y;

    //            Vector3 global_pos = this.server.GetGlobalLocation() + this.global_offset;

    //            this.characterTransform.position = global_pos;
    //            this.rootBone.position += global_pos; //* 1.7f;// * 1.7f;
    //            this.rootBone.rotation = Quaternion.Euler(qt.x, qt.y, qt.z);
    //        }
    //    }

    //}

    //Transform punchTargetTransform;
    public Transform GetPunchTargetTransform()
    {
        Transform punchTargetTransform;

        GameObject go = GameObject.Find("punch_target");
        punchTargetTransform = go.transform;
        Debug.Log("punch target: " + punchTargetTransform.position);
        //Debug.Log(punchTargetTransform.position);
        return punchTargetTransform;

    }

    // Update is called once per frame
    void Update()
    {

        if (Input.GetMouseButtonDown(0))
        {
            Debug.Log("Pressed primary button.");
            leftMousePressed = true;
            if (leftMousePressed)
            {
                punchInput punch_in = new punchInput();
                //animator = GetComponent<Animator>();
                //Vector3 r_hand_pos= animator.GetBoneTransform(HumanBodyBones.RightHand).position;

                Vector3 r_hand_pos = GameObject.Find("RightWrist_end").transform.position;
                Debug.Log(r_hand_pos);
                var punchTargetTransform = GetPunchTargetTransform();
                Vector3 target = punchTargetTransform.position;


                punch_in.hand = "left";
                punch_in.target_left = new double[] { target.x, target.y, target.z };
                punch_in.target_right = new double[] { r_hand_pos.x, r_hand_pos.y, r_hand_pos.z };

                var httpWebRequest = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:5000/punch_in");
                httpWebRequest.ContentType = "application/json";
                httpWebRequest.Method = "POST";

                string json = JsonUtility.ToJson(punch_in);

                using (var streamWriter = new StreamWriter(httpWebRequest.GetRequestStream()))
                {
                    streamWriter.Write(json);
                }

                var httpResponse = (HttpWebResponse)httpWebRequest.GetResponse();
                using (var streamReader = new StreamReader(httpResponse.GetResponseStream()))
                {
                    var result = streamReader.ReadToEnd();
                    Debug.Log(result);
                }

                leftMousePressed = false;
            }
            Debug.Log("Left");
        }

        if (Input.GetMouseButtonDown(1))
        {
            Debug.Log("Pressed secondary button.");
            rightMousePressed = true;

            if (rightMousePressed)
            {
                punchInput punch_in = new punchInput();
                //animator = GetComponent<Animator>();
                //Vector3 r_hand_pos= animator.GetBoneTransform(HumanBodyBones.RightHand).position;

                //Vector3 l_hand_pos = GameObject.Find("LefttWrist_end").transform.position;
                Vector3 l_hand_pos = GameObject.Find("RightWrist_end").transform.position;
                Debug.Log(l_hand_pos);
                var punchTargetTransform = GetPunchTargetTransform();
                Vector3 target = punchTargetTransform.position;


                punch_in.hand = "right";
                punch_in.target_left = new double[] { l_hand_pos.x, l_hand_pos.y, l_hand_pos.z };
                punch_in.target_right = new double[] { target.x, target.y, target.z };

                var httpWebRequest = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:5000/punch_in");
                httpWebRequest.ContentType = "application/json";
                httpWebRequest.Method = "POST";

                string json = JsonUtility.ToJson(punch_in);

                using (var streamWriter = new StreamWriter(httpWebRequest.GetRequestStream()))
                {
                    streamWriter.Write(json);
                }

                var httpResponse = (HttpWebResponse)httpWebRequest.GetResponse();
                using (var streamReader = new StreamReader(httpResponse.GetResponseStream()))
                {
                    //var result = streamReader.ReadToEnd();

                    string json_obj = streamReader.ReadToEnd();
                    string data_1 = JObject.Parse(json_obj)["status"].ToString();
                    string data_2 = JObject.Parse(json_obj)["pose"].ToString();

                    Debug.Log(data_1);
                    Debug.Log(data_2);

                    JObject o = JObject.Parse(json_obj);
                    var msg = JObject.Parse(json_obj)["message"];

                    Debug.Log("******************************");
                    Debug.Log(msg);

                    //foreach (var p in o.Properties())
                    //{
                    //    Debug.Log("name: " + p.Name);
                    //    Debug.Log("type: " + p.Value.Type);
                    //    Debug.Log("value: " + p.Value);
                    //}

                    //var contributorsAsJson = streamReader.ReadToEnd();
                    //var contributors = JsonConvert.DeserializeObject<List<Contributor>>(contributorsAsJson);
                    //contributors.ForEach(Debug.Log);


                    //Debug.Log(result);
                }

                rightMousePressed = false;
            }
            Debug.Log("Right");
        }

        if (Input.GetMouseButtonDown(2))
        {
            Debug.Log("Pressed middle click.");
            midMousePressed = true;
        }

    }

}
