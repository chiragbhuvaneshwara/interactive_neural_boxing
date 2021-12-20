///<summary>
/// XsLiveObject is a component to bind Xsens MVN Studio object stream to Unity3D Game Engine.
///
///</summary>
/// <version>
/// 1.0, 2020.09.29 by Rob Koster
/// </version>
///<remarks>
/// Copyright (c) 2020, Xsens Technologies B.V.
/// All rights reserved.
///
/// Redistribution and use in source and binary forms, with or without modification,
/// are permitted provided that the following conditions are met:
///
/// 	- Redistributions of source code must retain the above copyright notice,
///		  this list of conditions and the following disclaimer.
/// 	- Redistributions in binary form must reproduce the above copyright notice,
/// 	  this list of conditions and the following disclaimer in the documentation
/// 	  and/or other materials provided with the distribution.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
/// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
/// AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
/// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
/// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
/// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
/// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
/// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///</remarks>

using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;

namespace xsens
{

    /// <summary>
    /// Xsens Live Object.
    ///
    /// This class provide the logic to read an object pose from MVN Stream and
    /// retarget the animation to a GameObject.
    /// </summary>
    /// <remarks>
    /// Attach this component to the object and bind the MvnActors with this object.
    /// </remarks>
    public class XsLiveObject : MonoBehaviour
    {
        public XsStreamReader mvnActors;			//network streamer, which contains all 4 actors' poses
        public int objectID = 1;                    //current object ID, where 1 is the first streamed object from MVN
        public bool applyOriginalRotation = false;
        public Vector3 RotationOffset;
        private int actorID = 1;

        private Transform target;                   //Reference to the object in Unity3D.
        private Quaternion origRot;
        private bool isInited;						//flag to check if the plugin was correctly intialized.
        private int objectCount = 0;                //used to figure out the total segment count provided by the data

        /// <summary>
        /// Wake this instance and initialize the live objects.
        /// </summary>
        IEnumerator Start()
        {
            isInited = false;
            //save start positions
            target = gameObject.transform;
            origRot = target.rotation;

            // Search for the network stream, so we can communicate with it.
            if (mvnActors == null)
            {
                Debug.LogError("[xsens] No MvnActor found! You must assign an MvnActor to this component.");
                yield return null;
            }

            //Wait for data to come in so that we can figure out incomming segment counts before setup
            while (!mvnActors.objectsEstablished(out actorID, out objectCount))
            {
                yield return null;
            }

            isInited = true;
        }

        /// <summary>
        ///	Update the objects in every frame.
        ///
        /// The mvn object orientations and positions are read from the network,
        /// using the MvnLiveActor component.
        /// This component provides all data for the current pose for each object.
        /// </summary>
        void Update()
        {
            //only do magic if we have everything ready
            if (!isInited)
                return;

            Vector3[] latestPositions;
            Quaternion[] latestOrientations;
            // Get the pose data in one call, else the orientation might come from a different pose
            // than the position
            if (objectID <= objectCount && mvnActors.getLatestPose(actorID - 1, out latestPositions, out latestOrientations))
            {
                transform.position = latestPositions[objectID-1];
                // First apply a rotation of 90 degrees around the Y axis (this is the Z-axis in MVN) to make the rotation consistent with MVN
                // Then apply the quaternion roatation from the object in MVN
                // Finally apply the original rotation to make sure the object is oriented the same as modeled in Unity.
                transform.rotation = Quaternion.Euler(0.0f, -90.0f, 0.0f) * latestOrientations[objectID - 1] * Quaternion.Euler(RotationOffset.x, RotationOffset.y, RotationOffset.z);
                if (applyOriginalRotation)
                    transform.rotation *= origRot;
            }
        }
    }//class XsLiveObject
}//namespace Xsens