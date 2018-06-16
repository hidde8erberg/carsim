using System.Collections;
using UnityEngine;

public class Car : MonoBehaviour {

    public WheelCollider wheel_front_left;
    public WheelCollider wheel_front_right;
    public WheelCollider wheel_back_left;
    public WheelCollider wheel_back_right;

    private float max_steer_angle = 45f;
    private float motorTorque = 500f;

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
    }
}
