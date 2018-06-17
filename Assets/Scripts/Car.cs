using System.Collections;
using UnityEngine;

public class Car : MonoBehaviour {

    public WheelCollider wheel_front_left;
    public WheelCollider wheel_front_right;
    public WheelCollider wheel_back_left;
    public WheelCollider wheel_back_right;

    public MeshRenderer wheelmesh_front_left;
    public MeshRenderer wheelmesh_front_right;
    public MeshRenderer wheelmesh_back_left;
    public MeshRenderer wheelmesh_back_right;

    private float max_steer_angle = 30f;
    private float motorTorque = 100f;

    // Use this for initialization
    void Start () {
	
	}
	
	// Update is called once per frame
	void Update () {
	
	}
    
    void FixedUpdate () {
        drive();
        steer();
    }

    void drive () {
        wheel_front_left.motorTorque = motorTorque;
        wheel_front_right.motorTorque = motorTorque;
        //wheel_back_left.motorTorque = motorTorque;
        //wheel_back_right.motorTorque = motorTorque;
    }

    void steer () {
        float steer_angle = Input.GetAxis("Horizontal") * max_steer_angle;
        wheel_front_left.steerAngle = steer_angle;
        wheel_front_right.steerAngle = steer_angle;

        wheelmesh_front_left.transform.eulerAngles = new Vector3(0, steer_angle, 0);
        wheelmesh_front_right.transform.eulerAngles = new Vector3(0, steer_angle, 0);
    }
}
