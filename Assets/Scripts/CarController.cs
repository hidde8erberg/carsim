using UnityEngine;

public class CarController : MonoBehaviour {

    private float m_horizontalInput;
    private float m_verticalInput;
    private float m_steeringAngle;

    public WheelCollider frontLeftWheel, frontRightWheel;
    public WheelCollider rearLeftWheel, rearRightWheel;
    public Transform frontLeftTransform, frontRightTransform;
    public Transform rearLeftTransform, rearRightTransform;
    public float maxSteerAngle = 30;
    public float motorForce;

    private Vector3 start_pos;
    private Quaternion start_rot;

    public float TravelDist;
    private Vector3 last_pos;

    public void Start()
    {
        start_pos = transform.position;
        start_rot = transform.rotation;

        TravelDist = 0;
        last_pos = transform.position;
    }

    private void OnCollisionEnter()
    {
        GetComponent<Client>().collision = 1;
        
        transform.position = start_pos;
        transform.rotation = start_rot;

        Rigidbody rbody = GetComponent<Rigidbody>();
        rbody.velocity = new Vector3(0,0,0);
        rbody.angularVelocity = new Vector3(0,0,0);
        
        TravelDist = 0;
    }

    private void DistanceTravelled()
    {
        TravelDist += Vector3.Distance(transform.position, last_pos);
        last_pos = transform.position;
    }
    
    public void GetInput()
    {
        m_horizontalInput = Input.GetAxis("Horizontal");
        m_verticalInput = Input.GetAxis("Vertical");
    }

    private void Steer()
    {
        m_steeringAngle = maxSteerAngle * m_horizontalInput;
        frontLeftWheel.steerAngle = m_steeringAngle;
        frontRightWheel.steerAngle = m_steeringAngle;
    }

    private void Accelerate()
    {
		frontLeftWheel.motorTorque = m_verticalInput * motorForce;
		frontRightWheel.motorTorque = m_verticalInput * motorForce;
        // rearLeftWheel.motorTorque = m_verticalInput * motorForce;
        // rearRightWheel.motorTorque = m_verticalInput * motorForce;
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
        Vector3 _pos = _transform.position;
        Quaternion _quat = _transform.rotation;

        _collider.GetWorldPose(out _pos, out _quat);

        _transform.position = _pos;
        _transform.rotation = _quat;
    }

    private void FixedUpdate()
    {
        GetInput();
        Steer();
        Accelerate();
        UpdateWheelPoses();
        DistanceTravelled();
        Debug.Log(TravelDist);
    }

}