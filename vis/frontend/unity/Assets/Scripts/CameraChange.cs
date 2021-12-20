using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraChange : MonoBehaviour
{
    // Define Cams
    public Camera cam1;
    public Camera cam2;
    public Camera cam3;


    //Define List of Cameras
    private List<Camera> cameras = new List<Camera>();

    void Start()
    {
        // Add inspector Cameras here
        cameras.Add(cam1);
        cameras.Add(cam2);
        cameras.Add(cam3);
    }

    void Update()
    {
        // Check for input and swap accordingly
        if (Input.GetKeyDown(KeyCode.Y))
        {
            //Debug.Log("cam1");
            SwapCamera(cam1);
        };
        if (Input.GetKeyDown(KeyCode.X))
        {
            //Debug.Log("cam2");
            SwapCamera(cam2);
         };
        if (Input.GetKeyDown(KeyCode.C))
        {
            //Debug.Log("cam2");
            SwapCamera(cam3);
        };
    }

    public void SwapCamera(Camera cam)
    {
        // performance tradeoff
        foreach (Camera c in cameras)
        {
            c.enabled = false;
        }
        //Debug.Log(cam.name);
        cam.enabled = true;
    }
}
