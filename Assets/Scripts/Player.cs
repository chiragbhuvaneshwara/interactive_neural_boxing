using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using UnityEngine;
public class Player : MonoBehaviour
{
    [SerializeField] private Transform groundCheckTransform = null;
    private bool jumpPressed;
    private float horizontalInput;
    private Rigidbody RigidBodyComp;
    private bool isGrounded;
    private int supJumpCount;
    private bool leftMousePressed;
    private bool rightMousePressed;
    private bool midMousePressed;

    [System.Serializable]
    public class punchInput
    {
        public string hand;
        public double[] target_left;
        public double[] target_right;
    }

    // Start is called before the first frame update
    void Start()
    {
        RigidBodyComp = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {

        if (Input.GetMouseButtonDown(0))
        {
            Debug.Log("Pressed primary button.");
            leftMousePressed = true;
            Debug.Log("Left");
        }

        if (Input.GetMouseButtonDown(1))
        {
            Debug.Log("Pressed secondary button.");
            rightMousePressed = true;
        }

        if (Input.GetMouseButtonDown(2))
        {
            Debug.Log("Pressed middle click.");
            midMousePressed = true;
        }

        if (Input.GetKeyDown(KeyCode.Space))
        {
            jumpPressed = true;
            Debug.Log("J");
        }

        horizontalInput = Input.GetAxis("Horizontal");
    }

    private void FixedUpdate()
    {

        if (leftMousePressed)
        {
            punchInput punch_in = new punchInput();
            punch_in.hand = "left";
            punch_in.target_left = new double[] { -0.07795768, -1.259803896, -1.021392921 };
            punch_in.target_right = new double[] { 0.48645876, -4.699789, 1.029539222 };

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

        //if (jumpPressed)
        //{

        //    /////////////////////////////*****get_from_server*****/////////////////////////////

        //    //WebRequest request = WebRequest.Create(" http://127.0.0.1:5000/ ");
        //    //WebResponse response = request.GetResponse();

        //    //using (Stream dataStream = response.GetResponseStream())
        //    //{
        //    //    // Open the stream using a StreamReader for easy access.
        //    //    StreamReader reader = new StreamReader(dataStream);
        //    //    // Read the content.
        //    //    string responseFromServer = reader.ReadToEnd();
        //    //    // Display the content.
        //    //    Debug.Log(responseFromServer);
        //    //    //Console.WriteLine(responseFromServer);
        //    //}

    }
}
