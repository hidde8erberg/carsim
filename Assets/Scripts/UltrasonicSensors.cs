using UnityEngine;

public class UltrasonicSensors : MonoBehaviour
{
	public GameObject Start;
	public GameObject Car;

	public Material ShadowMaterial;

	[Range(0.1f, 100f)]
	public float DetectionDistance = 10f;
	
	[Range(1, 100)]
	public int Amount = 5;

    [Range(1, 360)]
    public int Angle = 140;

	public float DistanceWarning = 5f;
	public float DistanceDanger = 3f;
	
	public Color ColorSafe = new Color(0, 255, 0); 
	public Color ColorWarning = new Color(0, 0, 255); 
	public Color ColorDanger = new Color(255, 0, 0);

	public Line[] _lines;

	private void DoRaycast()
	{
		var positionStart = Start.transform.position;
		var alpha = Angle / (float) Amount;

		_lines = new Line[Amount];
		
		for (var i = 0; i < Amount; i++)
		{
			var offset = Mathf.Floor(Amount / 2f);
			offset -= Amount % 2 == 0 ? 0.5f : 0;

			var beta = alpha * (i - offset);
			
			var direction = Car.transform.rotation * (Quaternion.AngleAxis(beta, Vector3.down) * Vector3.forward);
			
			// Debug.Log("Angle " + beta + ": " + direction);

			RaycastHit hit;
			
			if (Physics.Raycast(positionStart, direction, out hit, DetectionDistance))
			{
				var distance = Vector3.Distance(positionStart, hit.point);
				
				_lines[i] = new Line(positionStart, direction, distance);
				// Debug.Log("Detected object: " + distance);
			}
			else
			{
				_lines[i] = new Line(positionStart, direction, null);
			}
		}
	}
	
	private void DrawLines()
	{
		GL.PushMatrix();
		ShadowMaterial.SetPass(0);
		GL.Begin(GL.LINES);
		
		foreach (var line in _lines)
		{
			Color drawColor;

			if (line.Distance == null || line.Distance > DistanceWarning)
			{
				drawColor = ColorSafe;
			}
			else if (line.Distance > DistanceDanger)
			{
				drawColor = ColorWarning;
			}
			else
			{
				drawColor = ColorDanger;
			}
			
			DrawLine(line.Origin, line.Direction, drawColor);
		}
		
		GL.End();
		GL.PopMatrix();
	}
	
	private void DrawLine(Vector3 origin, Vector3 direction, Color color) 
	{
		if (!Application.isPlaying)
		{
			Debug.DrawRay(origin, direction * DetectionDistance, color);
			return;
		}
		
		GL.Color(color);
		GL.Vertex(origin);
		GL.Vertex(origin + direction * DetectionDistance);
	}

	private void FixedUpdate()
	{
		DoRaycast();
	}
    /*
	private void OnDrawGizmos()
	{
		if (Application.isPlaying) return;
		
		DoRaycast();	
		DrawLines();
	}
    */
	private void OnPostRender()
	{
		DrawLines();
	}
    
	public class Line
	{
		public Vector3 Origin { get; private set; }
		public Vector3 Direction { get; private set; }
		public float? Distance { get; private set; }

		public Line(Vector3 origin, Vector3 direction, float? distance)
		{
			Origin = origin;
			Direction = direction;
			Distance = distance;
		}
	}
}
