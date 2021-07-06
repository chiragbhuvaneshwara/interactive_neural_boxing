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
    private int leftTrajReached;
    private int rightTrajReached;

    private bool leftMousePressed;
    private bool rightMousePressed;
    private bool midMousePressed;

    public Transform rootBone = null;
    public Transform characterTransform = null;

    private Dictionary<string, Bone> bonemap = new Dictionary<string, Bone>();

    public bool start = false;
    public bool fix_global_pos = false;
    public float velocity_scale = 1.0f;
    public int num_traj_pts;
    private Dictionary<string, List<GameObject>>Trajectory = new Dictionary<string, List<GameObject>>();
    private Dictionary<string, LineRenderer> TrajLineRenderer = new Dictionary<string, LineRenderer>();
    private List<string> TrajNames = new List<string> { "root", "right_wrist", "left_wrist"};
    //public List<GameObject> Trajectory;
    //public LineRenderer TrajLineRenderer;
    private Vector3 global_offset = Vector3.zero;

    //public MultiMotionServer server = null;
    public MultiMotionServer server = new MultiMotionServer();

    public int TrajInTargetRange(int num_traj_reached_target, string target, string dict_key)
    {

        //Debug.Log(num_traj_reached_target);
        Vector3 punch_target = GameObject.Find(target).transform.position;
        //Vector3 wrist_pos = GameObject.Find(wrist).transform.position;
        //int num_traj_reached_target = 0;
        for (var i = Mathf.RoundToInt(num_traj_pts/2)+1; i < num_traj_pts - num_traj_reached_target; i++) 
        { 
            var tr_go = Trajectory[dict_key][i];
            
            Vector3 tr_pos = tr_go.transform.position;
            //Debug.Log((punch_target - tr_pos).magnitude);
            if ((punch_target - tr_pos).magnitude < 0.1)
            {
                //Debug.Log(tr_go.name);
                num_traj_reached_target += 1;
            }
        }
        //Debug.Log(num_traj_reached_target);

        //if ((punch_target - tr_pos).magnitude < 0.25)
        //{
        //    return true;
        //}
        //else
        //{
        //    return false;
        //}

        return num_traj_reached_target;
    }    
    
    public bool WristInTargetRange(string target, string wrist, bool reverse=false)
    {

        Vector3 punch_target = GameObject.Find(target).transform.position;
        Vector3 wrist_pos = GameObject.Find(wrist).transform.position;


        //Debug.Log((punch_target - wrist_pos).magnitude);

        var diff_mag = 0.25;
        if (reverse){
            diff_mag = 0.11;
            punch_target += new Vector3(0, 0, (float)-0.2);
        }

        Debug.Log((punch_target - wrist_pos).magnitude);

        //if ((punch_target - wrist_pos).magnitude < 0.13) {
        if ((punch_target - wrist_pos).magnitude < diff_mag) {
            return true;
        }
        else
        {
            return false;
        }

    }

    private void initializeBones(Transform t)
    {
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
        this.global_offset += this.transform.position;
        server = new MultiMotionServer();
        this.initializeBones(this.rootBone);
        server.Start();
        num_traj_pts = 10;
        leftTrajReached = 0;
        rightTrajReached = 0;
        createTrajVisObjs("root", Color.black, num_traj_pts);
        createTrajVisObjs("right_wrist", Color.red, num_traj_pts);
        createTrajVisObjs("left_wrist", Color.blue, num_traj_pts);
    }

    private void createTrajVisObjs(string tr_name, Color col, int num_tr_pts)
    {
        ////////////////////////TRAJ_OBJS///////////////////////////////////////
        var traj_spheres = new List<GameObject>(num_tr_pts);
        var GameObjectScale = 0.01f;
        for (int i = 0; i < num_tr_pts; i++)
        {
            var point = GameObject.CreatePrimitive(PrimitiveType.Sphere);

            point.GetComponent<MeshRenderer>().material = new Material(Shader.Find("Sprites/Default"));
            point.GetComponent<MeshRenderer>().material.SetColor("Black", Color.black);
            point.name = "traj_" + tr_name + "_" + i.ToString();
            point.transform.localScale = new Vector3(GameObjectScale, GameObjectScale, GameObjectScale);
            traj_spheres.Add(point);
        }
        Trajectory.Add(tr_name, traj_spheres);

        Color c1 = col;
        Color c2 = new Color(1, 1, 1, 0);

        GameObject obj = new GameObject("line");
        LineRenderer traj_line;
        traj_line = obj.AddComponent<LineRenderer>();
        traj_line.name = tr_name;
        traj_line.widthMultiplier = GameObjectScale;
        traj_line.material = new Material(Shader.Find("Sprites/Default"));
        traj_line.positionCount = num_tr_pts;
        traj_line.startColor = c1;
        traj_line.endColor = c2;

        TrajLineRenderer.Add(tr_name, traj_line);
        ////////////////////////TRAJ_OBJS///////////////////////////////////////
    }

    private void updateTrajVisObjs(string target)
    {
        ////////////////////////TRAJ_OBJS///////////////////////////////////////
        var tr = Trajectory[target];
        for (int i = 0; i < num_traj_pts; i++)
        {
            //Trajectory[i].transform.position = this.server.GetTrPos(target, i);
            tr[i].transform.position = this.server.GetTrPos(target, i);
        }
        Trajectory[target] = tr;

        var tr_line = TrajLineRenderer[target];
        for (int i = 0; i < num_traj_pts; i++)
        {
            //TrajLineRenderer.SetPosition(i, Trajectory[i].transform.position);
            tr_line.SetPosition(i, Trajectory[target][i].transform.position);
        }
        TrajLineRenderer[target] = tr_line;
        ////////////////////////TRAJ_OBJS///////////////////////////////////////
    }

    private void processTransforms(Transform t)
    {
        
        if (t.name.Contains("end"))
        {
            // End condition: end joint of final bone reached
            return;
        }

        // apply transforms
        Vector3 pos = this.server.GetJointPosition(t.name);// * 1.7f;
        t.position = pos;

        if (t.childCount > 0 && !t.GetChild(0).name.Contains("end"))
        {
            int childID = 0;
            if (t.childCount > 1)
            {
                childID = 1;
            }
            Vector3 toChildPos = this.server.GetJointPosition(t.GetChild(childID).name) - t.position; //* 1.7f
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
        ////////////////////////TRAJ_OBJS///////////////////////////////////////
        foreach (string target in TrajNames) 
        {
            updateTrajVisObjs(target);
        }
        ////////////////////////TRAJ_OBJS///////////////////////////////////////
    }

    void LateUpdate()
    {
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
        List<float> dir = new List<float> { 0, 0 };
        if (Input.GetKey(KeyCode.A))
        {
            dir[0] = 1;
        }
        if (Input.GetKey(KeyCode.D))
        {
            dir[0] = -1;
        }
        if (Input.GetKey(KeyCode.W))
        {
            dir[1] = -1;
        }
        if (Input.GetKey(KeyCode.S))
        {
            dir[1] = 1;
        }

        var temp = new Vector2(dir[0], dir[1]).normalized;

        dir[0] = temp.x;
        dir[1] = temp.y;

        if (Input.GetMouseButton(0) || leftMousePressed)
        {
            if (Input.GetMouseButton(0))
                Debug.Break();

            leftMousePressed = true;
            var isWristInTargetRange = server.ManagedUpdate("left", dir, leftTrajReached);
            if (isWristInTargetRange) {
                Debug.Log("LEnd");
                leftMousePressed = false;
            }
        }
        else if (Input.GetMouseButton(1) || rightMousePressed)
        {
            if (Input.GetMouseButton(1))
                Debug.Break();

            rightMousePressed = true;
            var isWristInTargetRange = server.ManagedUpdate("right", dir, rightTrajReached);

            if (isWristInTargetRange) { 
                Debug.Log("REnd");
                rightMousePressed = false;
            }
        }
        else if (Input.GetMouseButton(2))
        {
            midMousePressed = true;
            midMousePressed = false;
        }
        else
        {
            bool status = server.ManagedUpdate("none", dir, 0);
            //server.ManagedUpdate("left", dir);
            //server.ManagedUpdate("left", dir, 0);
        }
    }

}
