The functionality of this repository is separated into different components:

* nn: contains implementations of neural network models
* controlers: contain motion controlers to generate motion for a specific user input. They are using nn implementations. 
* servers: Thrift server implementations create the bridge between the Game Engine (e.g. Unity) and the motion controlers. 

The basic idea is, to re-use as much code as possible. At the moment, the following structure could be achieved:

1 Datamodel - 1 Controler - 2 Server (single, multi-user) - N different network architectures trained for this datamodel (pfnn with np and tf backend, vinn)

