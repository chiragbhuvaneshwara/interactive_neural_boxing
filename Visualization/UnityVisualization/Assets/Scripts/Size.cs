using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Size : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        var renderer = GetComponent <Renderer> ();
        var bound = renderer.bounds.size;
        Debug.Log("Hello");
        Debug.Log(bound);
    }

    // Update is called once per frame
    void Update()
    {
    }
}
