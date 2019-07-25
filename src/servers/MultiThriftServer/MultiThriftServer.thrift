namespace csharp MultiMosiServer
namespace py MultiMosiServer
namespace cpp MultiMosiServer

// Data field for a posture
struct TPosture
{
	1: required list<TBone> bones;
	2: required map<string,i32> bone_map;
	3: required TVector3 location;
	4: required double rotation;
}

// Data field for a single bone
struct TBone
{
	1: required string name;
	2: required TVector3 position;
	3: required TQuaternion rotation;
	4: optional list<string> children;
	5: optional string parent; // child? // Unity bone mapping required!
}

// Vector3 
struct TVector3
{
	1: required double x;
	2: required double y;
	3: required double z;
}

// Quaternion:
struct TQuaternion
{
	1: required double w;
	2: required double x;
	3: required double y;
	4: required double z;
}

// Gait
enum TGait
{
	standing,
	walking,
	running
}

// a simple motion server interface requiring a direction as an input
service T_multi_directional_motion_server
{
	i32 registerSession(),
	TPosture getZeroPosture(),
	TPosture fetchFrame(1: double time, 2: TPosture currentPosture, 3: TVector3 direction, 4: TGait gait, 5: i32 session_id)
}