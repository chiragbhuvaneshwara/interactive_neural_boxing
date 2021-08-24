using System.Collections.Generic;
using UnityEngine;
using MultiMosiServer;
using System;




public partial class Player : MonoBehaviour
{
    private int leftTrajReached;
    private int rightTrajReached;

    private bool leftMousePressed;
    private bool rightMousePressed;
    private bool midMousePressed;

    private List<float> facing_dir = new List<float> { 0, 1 };

    public Transform rootBone = null;
    public Transform characterTransform = null;

    private Dictionary<string, Bone> bonemap = new Dictionary<string, Bone>();

    public bool start = false;
    public bool eval = false;
    public bool fix_global_pos = false;
    public float velocity_scale = 1.0f;
    public int num_traj_pts;
    private Dictionary<string, List<GameObject>>Trajectory = new Dictionary<string, List<GameObject>>();
    private Dictionary<string, LineRenderer> TrajLineRenderer = new Dictionary<string, LineRenderer>();
    private List<string> TrajNames = new List<string> { "root", "right_wrist", "left_wrist"};
    private Vector3 global_offset = Vector3.zero;

    public MultiMotionServer server = new MultiMotionServer();

    

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
    }

    private void updateTrajVisObjs(string target)
    {
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
        foreach (string target in TrajNames) 
        {
            updateTrajVisObjs(target);
        }
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

    private float GetRandomFloat(System.Random random, double min, double max)
    {
        return Convert.ToSingle(min + (random.NextDouble() * (max - min)));
    }

    public void UpdatePunchTargetPosition()
    {

        GameObject go = GameObject.Find("punch_target");
        var random = new System.Random();
        var x = GetRandomFloat(random, -0.4, 0.4);
        var y = GetRandomFloat(random, 1.45, 1.9);
        var z = GetRandomFloat(random, 0.3, 0.6);

        go.transform.position = new Vector3(x, y, z);
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
            dir[1] = 1;
        }
        if (Input.GetKey(KeyCode.S))
        {
            dir[1] = -1;
        }

        var temp = new Vector2(dir[0], dir[1]).normalized;
        //var temp = new Vector2(dir[0], dir[1]);

        dir[0] = temp.x;
        dir[1] = temp.y;
        
        
        if (Input.GetKey(KeyCode.H))
        {
            facing_dir = new List<float> { 1, 0 } ;
        }
        if (Input.GetKey(KeyCode.K))
        {
            facing_dir = new List<float> { -1, 0 } ;
        }
        if (Input.GetKey(KeyCode.U))
        {
            facing_dir = new List<float> { 0, 1 } ;
        }
        if (Input.GetKey(KeyCode.J))
        {
            facing_dir = new List<float> { 0, -1 } ;
        }

        var facing_dir_vec = new Vector2(facing_dir[0], facing_dir[1]).normalized;

        facing_dir[0] = facing_dir_vec.x;
        facing_dir[1] = facing_dir_vec.y;

        if (Input.GetMouseButton(0) || leftMousePressed)
        {
            //if (Input.GetMouseButton(0))
                //Debug.Break();

            leftMousePressed = true;
            var isWristInTargetRange = server.ManagedUpdate("left", dir, facing_dir, leftTrajReached, false);
            if (isWristInTargetRange) {
                Debug.Log("LEnd");
                leftMousePressed = false;
            }
        }
        else if (Input.GetMouseButton(1) || rightMousePressed)
        {
            //if (Input.GetMouseButton(1))
                //Debug.Break();

            rightMousePressed = true;
            var isWristInTargetRange = server.ManagedUpdate("right", dir, facing_dir, rightTrajReached, false);

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
            bool status = server.ManagedUpdate("none", dir, facing_dir, 0, eval);
            //if (eval & status) {
                //UpdatePunchTargetPosition();
            //}
        }
    }

}
