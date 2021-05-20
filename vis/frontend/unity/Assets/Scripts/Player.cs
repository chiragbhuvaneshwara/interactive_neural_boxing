﻿using System.Collections;
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
    public int num_tr_pts = 10;
    private Dictionary<string, List<GameObject>>Trajectory = new Dictionary<string, List<GameObject>>();
    private Dictionary<string, LineRenderer> TrajLineRenderer = new Dictionary<string, LineRenderer>();
    private List<string> TrajNames = new List<string> { "root", "right_wrist", "left_wrist"};
    //public List<GameObject> Trajectory;
    //public LineRenderer TrajLineRenderer;
    private Vector3 global_offset = Vector3.zero;

    //public MultiMotionServer server = null;
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

        createTrajVisObjs("root", Color.black);
        createTrajVisObjs("right_wrist", Color.red);
        createTrajVisObjs("left_wrist", Color.blue);
    }

    private void createTrajVisObjs(string tr_name, Color col)
    {
        ////////////////////////TRAJ_OBJS///////////////////////////////////////
        var traj_spheres = new List<GameObject>(num_tr_pts);
        var GameObjectScale = 0.01f;
        for (int i = 0; i < num_tr_pts; i++)
        {
            var point = GameObject.CreatePrimitive(PrimitiveType.Sphere);

            point.GetComponent<MeshRenderer>().material = new Material(Shader.Find("Sprites/Default"));
            point.GetComponent<MeshRenderer>().material.SetColor("Black", Color.black);
            point.name = "tr_" + tr_name + "_" + i.ToString();
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
        for (int i = 0; i < num_tr_pts; i++)
        {
            //Trajectory[i].transform.position = this.server.GetTrPos(target, i);
            tr[i].transform.position = this.server.GetTrPos(target, i);
        }
        Trajectory[target] = tr;

        var tr_line = TrajLineRenderer[target];
        for (int i = 0; i < num_tr_pts; i++)
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
        int numActiveHorizDirKeys = 0;
        int numActiveVertDirKeys = 0;
        if (Input.GetKey(KeyCode.A))
        {
            //dir = new List<float> { -1, 0 };
            numActiveHorizDirKeys += 1;
            dir[0] += 1;
            //server.ManagedUpdate("none", dir);
        }
        if (Input.GetKey(KeyCode.D))
        {
            //dir = new List<float> { 1, 0 };
            numActiveHorizDirKeys += 1;
            dir[0] += -1;
            //server.ManagedUpdate("none", dir);
        }
        if (Input.GetKey(KeyCode.W))
        {
            //dir = new List<float> { 0, 1 };
            numActiveVertDirKeys += 1;
            dir[1] += 1;
            //server.ManagedUpdate("none", dir);
        }
        if (Input.GetKey(KeyCode.S))
        {
            //dir = new List<float> { 0, -1 };
            numActiveVertDirKeys += 1;
            dir[1] += -1;
            //server.ManagedUpdate("none", dir);
        }

        if (numActiveVertDirKeys==1 && numActiveVertDirKeys == 1)
        {
            dir[0] = dir[0] / 2;
            dir[1] = dir[1] / 2;
        }
        
        if (Input.GetMouseButton(0))
        //if (Input.GetMouseButtonDown(0))
        {
            //leftMousePressed = true;
            //if (leftMousePressed)
            //{
            server.ManagedUpdate("left", dir);
            //    leftMousePressed = false;
            //}
        }
        else if (Input.GetMouseButton(1))
        //else if (Input.GetMouseButtonDown(1))
        {
            //rightMousePressed = true;
            //if (rightMousePressed)
            //{
            server.ManagedUpdate("right", dir);
            //rightMousePressed = false;
            //}
        }
        else if (Input.GetMouseButton(2))
        {
            midMousePressed = true;
            midMousePressed = false;
        }
        else
        {
            //dir = new List<float> { 0, 0 };
            server.ManagedUpdate("none", dir);
        }
    }
}
