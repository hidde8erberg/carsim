using UnityEngine;
using System.Collections;

public class CameraFollow : MonoBehaviour {

    public Transform target;
    public float smoothSpeed = 0.125f;

    void FixedUpdate () {

        Vector3 smoothPos = Vector3.Lerp(transform.position, transform.position, smoothSpeed);
        transform.position = smoothPos;

    }
}
