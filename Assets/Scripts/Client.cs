using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Text;
using UnityEngine;

public class Client : MonoBehaviour {

    private static readonly UdpClient UdpClient = new UdpClient();
    private static readonly int PORT = 11111;

    void Start () {
        UdpClient.Client.Bind(new IPEndPoint(IPAddress.Any, PORT));
    }

    void FixedUpdate() {
        var lines = GameObject.Find("Main Camera").GetComponent<UltrasonicSensors>()._lines;
        var distances = new float[5];
        for (var i = 0; i < lines.Length; i++) {
            distances[i] = lines[i].Distance.HasValue ? (float)lines[i].Distance : 0f;
        }

        Array.Reverse(distances);

        var bytearray = new byte[distances.Length * 4];
        Buffer.BlockCopy(distances, 0, bytearray, 0, bytearray.Length);

        UdpClient.Send(bytearray, bytearray.Length, "127.0.0.1", PORT);

        lines = null;
        distances = null;
        
    }
}
