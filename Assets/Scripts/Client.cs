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
        var distances = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var bytearray = new byte[distances.Length * 4];
        Buffer.BlockCopy(distances, 0, bytearray, 0, bytearray.Length);

        // var data = Encoding.UTF8.GetBytes(distances);
        UdpClient.Send(bytearray, bytearray.Length, "127.0.0.1", PORT);
        
    }
}
