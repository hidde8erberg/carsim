using UnityEngine;
using System.Collections;
using UnityEngine.EventSystems;
using UnityEngine.Serialization;

public class UltrasonicSensors : MonoBehaviour
{
	public GameObject Start;
	public GameObject Car;

	[Range(0.1f, 100f)]
	public float DetectionDistance = 10f;
	
	[Range(1, 100)]
	public int Amount = 10;

	public float DistanceWarning = 5f;
	public float DistanceDanger = 3f;
	
	public Color ColorSafe = new Color(0, 255, 0); 
	public Color ColorWarning = new Color(0, 0, 255); 
	public Color ColorDanger = new Color(255, 0, 0);

	private void DrawLines()
	{
		var positionStart = Start.transform.position;
		var alpha = 360 / Amount;
		
		
		for (var i = 0; i <= Amount; i++)
		{
			var beta = alpha * i;
			var direction = Car.transform.rotation * (Quaternion.AngleAxis(beta, Vector3.down) * Vector3.forward);

			RaycastHit hit;
			
			if (Physics.Raycast(positionStart, direction, out hit, DetectionDistance))
			{
				var distance = Vector3.Distance(positionStart, hit.transform.position);
				
				Color drawColor;
				if (distance <= DistanceDanger)
				{
					drawColor = ColorDanger;
				} 
				else if (distance <= DistanceWarning)
				{
					drawColor = ColorWarning;
				}
				else
				{
					drawColor = ColorSafe;
				}
				
				Debug.DrawRay(positionStart, direction * DetectionDistance, drawColor);
				Debug.Log("Detected object: " + distance);
			}
			else
			{
				Debug.DrawRay(positionStart, direction * DetectionDistance, ColorSafe);
			}

		}
	}

	private void OnPostRender()
	{
		DrawLines();
	}

	private void OnDrawGizmos()
	{
		DrawLines();
	}
}
