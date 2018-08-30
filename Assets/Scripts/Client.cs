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

	void FixedUpdate () {
        float input = Input.GetAxis("Vertical");
        if (input != 0)
        {
            var data = Encoding.UTF8.GetBytes("GOOO");
            UdpClient.Send(data, data.Length, "127.0.0.1", PORT);
        }
    }
}
