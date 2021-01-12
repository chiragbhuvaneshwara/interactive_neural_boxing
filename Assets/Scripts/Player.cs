using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using UnityEngine;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using MultiMosiServer;

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

    //public MultiMotionServer server = null;
    public MultiMotionServer server = new MultiMotionServer();

    private void initializeBones(Transform t)
    {
        //TODO: update for Xsens Awinda skeleton
        if (t.name.Contains("end"))
        {
            // End condition: end joint of final bone reached
            return;
        }

        // apply transforms
        Quaternion startInverseRotation = new Quaternion(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w);
        int childID = 0;
        if (t.childCount > 1)
        {
            childID = 1;
        }
        Vector3 defaultDirection = new Vector3(0, 1, 0);
        if (t.childCount > 0)
        {
            Vector3 theChildPosition = t.GetChild(childID).transform.position;
            Vector3 thePosition = t.position;
            defaultDirection = (theChildPosition - thePosition);
        }

        Bone b = new Bone(startInverseRotation, defaultDirection);

        this.bonemap.Add(t.name, b);

        // Iteration step
        for (int i = 0; i < t.childCount; i++)
        {
            initializeBones(t.GetChild(i));
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        //RigidBodyComp = GetComponent<Rigidbody>();
        this.global_offset += this.transform.position;
        server = new MultiMotionServer();
        this.initializeBones(this.rootBone);
        server.Start();
    }

    private void processTransforms(Transform t)
    {
        
        //TODO: update for Xsens Awinda skeleton
        if (t.name.Contains("end"))
        {
            // End condition: end joint of final bone reached
            return;
        }

        // apply transforms
        Vector3 pos = this.server.GetJointPosition(t.name);// * 1.7f;  //TODO: update for your code
        t.position = pos;

        if (t.childCount > 0 && !t.GetChild(0).name.Contains("end"))
        {
            int childID = 0;
            if (t.childCount > 1)
            {
                childID = 1;
            }
            Vector3 toChildPos = this.server.GetJointPosition(t.GetChild(childID).name) - t.position; //* 1.7f //TODO: update for your code
            t.rotation = Quaternion.FromToRotation(this.bonemap[t.name].direction, toChildPos) * this.bonemap[t.name].rotation;


        }


        // Iteration step
        for (int i = 0; i < t.childCount; i++)
        {
            processTransforms(t.GetChild(i));
        }
    }

    void LateUpdate()
    {
        //TODO: update processTransforms to obtain relevant joint positions from your code

        if (start)
        {
            this.rootBone.rotation = Quaternion.identity;
            processTransforms(this.rootBone);

            if (!this.fix_global_pos)
            {
                Vector3 q = this.server.GetGlobalRotation().eulerAngles;
                Vector3 qt = this.rootBone.rotation.eulerAngles;
                qt.y = -q.y;

                Vector3 global_pos = this.server.GetGlobalLocation() + this.global_offset;

                this.characterTransform.position = global_pos;
                this.rootBone.position += global_pos; //* 1.7f;// * 1.7f;
                this.rootBone.rotation = Quaternion.Euler(qt.x, qt.y, qt.z);
            }
        }

    }


    // Update is called once per frame
    void Update()
    {

        if (Input.GetMouseButtonDown(0))
        {
            Debug.Log("Pressed primary button i.e left");
            leftMousePressed = true;
            if (leftMousePressed)
            {
                //string json_obj = server.UpdatePunchTarget("left");
                //MultiMotionServer.TPosture json_obj = server.UpdatePunchTargetFetchPosture("left");
                server.ManagedUpdate("left");

                leftMousePressed = false;
            }

        }

        if (Input.GetMouseButtonDown(1))
        {
            Debug.Log("Pressed secondary button i.e right");
            rightMousePressed = true;

            if (rightMousePressed)
            {
                //string json_obj = server.UpdatePunchTarget("right");
                server.ManagedUpdate("right");

                rightMousePressed = false;
            }

        }

        if (Input.GetMouseButtonDown(2))
        {
            Debug.Log("Pressed middle click.");
            midMousePressed = true;
        }

    }

}
