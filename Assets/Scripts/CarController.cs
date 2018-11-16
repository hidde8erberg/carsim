using UnityEngine;
using UnityEngine.UI;

public class CarController : MonoBehaviour {

    // private float m_horizontalInput;
    private float m_steeringAngle;

    public WheelCollider frontLeftWheel, frontRightWheel;
    public WheelCollider rearLeftWheel, rearRightWheel;
    public Transform frontLeftTransform, frontRightTransform;
    public Transform rearLeftTransform, rearRightTransform;
    public float maxSteerAngle = 25f;
    public float motorForce;
    public float acceleration = 0.4f;
    public float maxSpeed = 2f;
    public Text acc_text;
    public Text speed_text;

    private Vector3 start_pos;
    private Quaternion start_rot;
    private Vector3 last_pos;
    private bool _collision;
    
    [HideInInspector]
    public float TravelDist;
    public static float SteerInput;

    public void Start()
    {   
        start_pos = transform.position;
        start_rot = transform.rotation;

        TravelDist = 0;
        last_pos = transform.position;

        SteerInput = 0;
    }

    private void OnCollisionEnter()
    {
        GetComponent<Client>().collision = 1;
        
        transform.position = start_pos;
        transform.rotation = start_rot;

        var rbody = GetComponent<Rigidbody>();
        rbody.velocity = new Vector3(0,0,0);
        rbody.angularVelocity = new Vector3(0,0,0);
        
        TravelDist = 0;
        last_pos = start_pos;
        SteerInput = 0;
    }

    private void DistanceTravelled()
    {
        TravelDist += Vector3.Distance(transform.position, last_pos);
        last_pos = transform.position;
    }

    private void Steer()
    {
        // m_steeringAngle = maxSteerAngle * GetComponent<Server>().steerInput;
        m_steeringAngle = maxSteerAngle * SteerInput;
        frontLeftWheel.steerAngle = m_steeringAngle;
        frontRightWheel.steerAngle = m_steeringAngle;
    }

    private void Accelerate()
    {
        var rbody = GetComponent<Rigidbody>();
        var mag = rbody.velocity.magnitude;
        
        if (mag < maxSpeed)
        {
            frontLeftWheel.motorTorque = acceleration * motorForce;
            frontRightWheel.motorTorque = acceleration * motorForce;
            // rearLeftWheel.motorTorque = m_verticalInput * motorForce;
            // rearRightWheel.motorTorque = m_verticalInput * motorForce;
        }
    }

    private void UpdateWheelPoses()
    {
        UpdateWheelPose(frontLeftWheel, frontLeftTransform);
        UpdateWheelPose(frontRightWheel, frontRightTransform);
        UpdateWheelPose(rearLeftWheel, rearLeftTransform);
        UpdateWheelPose(rearRightWheel, rearRightTransform);
    }

    private void UpdateWheelPose(WheelCollider _collider, Transform _transform)
    {
        var _pos = _transform.position;
        var _quat = _transform.rotation;

        _collider.GetWorldPose(out _pos, out _quat);

        _transform.position = _pos;
        _transform.rotation = _quat;
    }

    private void FixedUpdate()
    {
        Steer();
        Accelerate();
        UpdateWheelPoses();
        DistanceTravelled();
        
        acc_text.text = "Acceleration: " + acceleration;
        speed_text.text = "Speed Lock: " + maxSpeed;
    }
    
    public void SetAcceleration(float input)
    {
        acceleration = input;
    }
    
    public void SetMaxspeed(float input)
    {
        maxSpeed = input;
    }
}