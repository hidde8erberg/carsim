using System;
using System.Net;
using System.Net.Sockets;
using UnityEngine;

public class Client : MonoBehaviour {

    private static readonly UdpClient UdpClient = new UdpClient();
    private static readonly int PORT = 11111;
    
    [HideInInspector]
    public int collision;

    void Start () {
        try 
        {
            UdpClient.Client.Bind(new IPEndPoint(IPAddress.Any, PORT));
            collision = 0;
        } 
        catch (Exception e) 
        {
            Debug.Log(e);
            throw;
        }
    }

    void Update() {
        try
        {   
            var lines = GameObject.Find("Main Camera").GetComponent<UltrasonicSensors>()._lines;
            var distances = new float[lines.Length + 2];
            for (var i = 0; i < lines.Length; i++) {
                distances[i] = lines[i].Distance.HasValue ? (float)lines[i].Distance : 0f;
            }

            Array.Reverse(distances);

            distances[lines.Length] = GetComponent<CarController>().TravelDist;
            distances[lines.Length + 1] = collision;

            var bytearray = new byte[distances.Length * 4];
            Buffer.BlockCopy(distances, 0, bytearray, 0, bytearray.Length);

            UdpClient.Send(bytearray, bytearray.Length, "127.0.0.1", PORT);

            lines = null;
            distances = null;
            collision = 0;
        }
        catch (Exception e)
        {
            Debug.Log(e);
            throw;
        }
    }
}
