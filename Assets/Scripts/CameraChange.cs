using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraChange : MonoBehaviour
{
    // Define Cams
    public Camera cam1;
    public Camera cam2;


    //Define List of Cameras
    private List<Camera> cameras = new List<Camera>();

    void Start()
    {
        // Add inspector Cameras here
        cameras.Add(cam1);
        cameras.Add(cam2);
    }

    void Update()
    {
        // Check for input and swap accordingly
        if (Input.GetKeyDown(KeyCode.H))
        {
        };
        if (Input.GetKeyDown(KeyCode.J))
        { 
            SwapCamera(cam2);
         };
    }

    public void SwapCamera(Camera cam)
    {
        // performance tradeoff
        foreach (Camera c in cameras)
        {
            c.enabled = false;
        }
        cam.enabled = true;
    }
}
