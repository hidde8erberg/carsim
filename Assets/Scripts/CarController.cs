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
    }

}