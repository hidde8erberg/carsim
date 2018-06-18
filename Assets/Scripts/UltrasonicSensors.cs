using UnityEngine;
using System.Collections;

public class UltrasonicSensors : MonoBehaviour {

    Color color = new Color(0.2F, 0.3F, 0.4F, 0.5F);
    public Vector3 offset;
    public float distance = 10f;

    void drawLine (Vector3 start)
    {
        GameObject line = new GameObject();
        line.transform.position = start;
        line.AddComponent<LineRenderer>();
        LineRenderer lr = line.GetComponent<LineRenderer>();
        // lr.material = new Material(Shader.Find("Clean Concrete Ground/Concrete_Ground_psd"));
        lr.SetColors(color, color);
        lr.SetWidth(0.1f, 0.1f);
        lr.SetPosition(0, start);
        Vector3 end = start + new Vector3(0,0,distance);
        Transform carRot = GetComponent<Transform>();
        Debug.Log(carRot.rotation.y);
        lr.SetPosition(1, Vector3.Scale(end, new Vector3(carRot.rotation.y, 0, 0)));
        //lr.transform.rotation = carRot.rotation;
        GameObject.Destroy(line, Time.deltaTime);
    }

    void FixedUpdate ()
    {
        drawLine(GetComponent<Transform>().position + offset);
    }
	
}
