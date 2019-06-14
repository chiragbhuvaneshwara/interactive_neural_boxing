namespace csharp SimpleMosiServer
namespace py SimpleMosiServer

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
	2: required TVector3 Position;
	3: optional list<string>children;
	4: optional string parent; // child? // Unity bone mapping required!
}

// Vector3 
struct TVector3
{
	1: required double X;
	2: required double Y;
	3: required double Z;
}

// Gait
enum TGait
{
	standing,
	walking,
	running
}

// a simple motion server interface requiring a direction as an input
service T_simple_directional_motion_server
{
	TPosture getZeroPosture(),
	TPosture fetchFrame(1: double time, 2: TPosture currentPosture, 3: TVector3 direction, 4: TGait gait)
}