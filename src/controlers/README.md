This folder contains implementations for neural network controllers. 

A controller maintains the state of the avatar and of the control variables. It prepares the neural network input based on this state and the "user input". 
Basic usage of the controller: 

1. pre_render - with input. Prepare the next state, including prediction
2. getPose - get the joint position in local space
3. getWorldPosRot - get root position and rotation in global space
4. postRender - clean-up / blending for preparation of next step. 

Currently, we have implemented a directional controller. 
Additional helper classes are Character and Trajectory to maintain the state of both entities and to provide basic functionality to operate with these. 