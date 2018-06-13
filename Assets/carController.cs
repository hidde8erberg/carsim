using UnityEngine;
using System.Collections;

public class carController : MonoBehaviour {

	float speedForce = 40f;
	float torqueForce = -100f;
	float driftFactor = 0.25f;

	// Use this for initialization
	void Start () {
	
	}

	// Update is called once per frame
	void Update () {

	}
	
	void FixedUpdate () {

		Rigidbody2D rb = GetComponent<Rigidbody2D>();

		rb.velocity = ForwardVelocity() + RightVelocity()*driftFactor;

        float tf = Mathf.Lerp(0, torqueForce, rb.velocity.magnitude / 5);

        if (Input.GetButton("Accelerate")) {
			rb.AddForce(transform.up * speedForce);
		}

        rb.angularVelocity = Input.GetAxis("Horizontal") * tf * (rb.velocity.magnitude / 10f);

        if (Input.GetButton("Brake")) {
			rb.AddForce(-transform.up * speedForce);
            rb.angularVelocity = Input.GetAxis("Horizontal") * -tf * (rb.velocity.magnitude / 10f);
        }

        /*if (rb.velocity[1] > 0) {
			rb.angularVelocity = Input.GetAxis("Horizontal") * tf * (rb.velocity.magnitude / 10f);
		}else if (rb.velocity[1] < 0) {
			rb.angularVelocity = Input.GetAxis("Horizontal") * -tf * (rb.velocity.magnitude / 10f);
		}*/

    }

	Vector2 ForwardVelocity() {
		return transform.up * Vector2.Dot(GetComponent<Rigidbody2D>().velocity, transform.up);
	}

	Vector2 RightVelocity() {
		return transform.right * Vector2.Dot(GetComponent<Rigidbody2D>().velocity, transform.right);
	}

}
