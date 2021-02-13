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
    public List<UnityEngine.GameObject> Trajectory;
    public List<UnityEngine.LineRenderer> TrajLines;
    public UnityEngine.LineRenderer TrajLineRenderer;

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
        Trajectory = new List<GameObject>(10);
        //TrajLines = new List<LineRenderer>(10);

        Material[] materials = Resources.FindObjectsOfTypeAll<Material>();
        Material red = null, green = null;
        foreach (Material t in materials)
        {
            if (t.name == "red") red = t;
            if (t.name == "green") green = t;
        }

        for (int i = 0; i < 10; i++)
        {
            var point = GameObject.CreatePrimitive(PrimitiveType.Sphere);

            point.GetComponent<MeshRenderer>().material = red;
            point.name = "tr_" + i.ToString();
            point.transform.localScale = new Vector3(0.1f, 0.1f, 0.1f);
            Trajectory.Add(point);

            //LineRenderer lineRenderer = gameObject.AddComponent<LineRenderer>();
            //lineRenderer.name = "tr_line_" + i.ToString();
            //TrajLines.Add(lineRenderer);
        }

        TrajLineRenderer = gameObject.AddComponent<LineRenderer>();
        TrajLineRenderer.name = "tr_line";
        //TrajLineRenderer.widthMultiplier = 0.01f;
        TrajLineRenderer.positionCount = 10;
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

    private void processTraj()
    {
        for (int i = 0; i < 10; i++)
        {
            //this.server.GetArmTrPos("Right", i);
            Trajectory[i].transform.position = this.server.GetArmTrPos("Right", i);
        }

        for (int i = 0; i < 10; i++)
        {
            //this.server.GetArmTrPos("Right", i);
            //TrajLines[i].SetPosition(i, new Vector3(i * 0.5f, Mathf.Sin(i + t), 0.0f));
            TrajLineRenderer.SetPosition(i, Trajectory[i].transform.position);
        }

    }

    void LateUpdate()
    {
        //TODO: update processTransforms to obtain relevant joint positions from your code

        if (start)
        {
            this.rootBone.rotation = Quaternion.identity;
            processTransforms(this.rootBone);
            processTraj();

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

        if (Input.GetMouseButton(0))
        {
            //Debug.Log("Pressed primary button i.e left");
            leftMousePressed = true;
            if (leftMousePressed)
            {
                //string json_obj = server.UpdatePunchTarget("left");
                //MultiMotionServer.TPosture json_obj = server.UpdatePunchTargetFetchPosture("left");
                server.ManagedUpdate("left");

                leftMousePressed = false;
            }

        }

        else if (Input.GetMouseButton(1))
        {
            //Debug.Log("Pressed secondary button i.e right");
            rightMousePressed = true;

            if (rightMousePressed)
            {
                //string json_obj = server.UpdatePunchTarget("right");
                server.ManagedUpdate("right");

                rightMousePressed = false;
            }

        }

        else if (Input.GetMouseButton(2))
        {
            midMousePressed = true;
            //Debug.Log("Pressed middle click.");
            midMousePressed = false;
        }

        else
        {
            //Debug.Log("No target");
            server.ManagedUpdate("right");
        }

    }

}
